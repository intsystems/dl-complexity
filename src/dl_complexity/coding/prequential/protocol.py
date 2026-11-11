"""Explicit causal preparation for finite categorical observation sequences.

The estimators only evaluate prepared ledgers. These functions perform the
separate training procedure. They pass labels to training/update callbacks only
after those labels have been encoded. Arbitrary user callbacks must honor this
contract; their closed-over state cannot be audited automatically.

Reference: Bornschein, Li & Hutter, arXiv:2210.07931, Section 2 and Appendix B.1.
Implemented variants: chunk-from-scratch, online minibatch, and uniform replay
from the revealed prefix. Replay streams/EMA/image augmentation are not claimed.
"""

import math
import weakref
from collections.abc import Callable, Sequence

import torch
from torch.nn import functional as F

from ..._validation import positive_int, scalar
from ...errors import InputError
from ..inputs import PrequentialEntry, PrequentialLedger


def _sequence(x, y, n_classes):
    r"""Validate finite categorical sequence shapes and the declared class alphabet.

    Args:
        x (Tensor): Observations [n, ...], kept by reference.
        y (LongTensor): Labels [n] within [0, n_classes).
        n_classes (int): Fixed alphabet size, at least two.

    Returns:
        int: Observation count n.

    Raises:
        InputError: If types, shapes or label bounds are invalid.

    Notes:
        O(n) label-range validation and O(n) temporary comparisons; no model call.
    """
    positive_int(n_classes, "n_classes")
    if n_classes < 2:
        raise InputError("n_classes must be declared and at least two")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise InputError("Preparation currently accepts input and target tensors")
    if x.ndim == 0 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise InputError("Inputs and labels need the same nonempty observation axis")
    if y.dtype != torch.long or bool(((y < 0) | (y >= n_classes)).any()):
        raise InputError(
            "Targets must be LongTensor class indices in the declared alphabet"
        )
    return len(y)


def _logits(state, x, predict_logits, n_classes):
    r"""Run detached eval predictions and restore original per-module mode flags.

    Args:
        state (object): Callable predictor or callback-owned learner state.
        x (Tensor): A currently scored batch.
        predict_logits (callable, optional): Explicit (state, x) prediction adapter.
        n_classes (int): Expected final logit dimension.

    Returns:
        Tensor: Detached finite [batch, n_classes] logits.

    Raises:
        InputError: If predictor output violates the categorical contract.

    Notes:
        Costs one forward F(batch) plus O(batch * classes) validation. For a Module
        the mode bookkeeping is O(number of modules). Buffers and arbitrary Python
        side effects are not rolled back; callers own the preparation state.
    """
    predictor = state if predict_logits is None else lambda z: predict_logits(state, z)
    if not callable(predictor):
        raise InputError("Provide a callable model or predict_logits(state, x)")
    flags = (
        [(module, module.training) for module in state.modules()]
        if isinstance(state, torch.nn.Module)
        else []
    )
    try:
        if flags:
            state.eval()
        with torch.no_grad():
            logits = predictor(x)
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise InputError("Predictor must return floating logits")
        if logits.shape != (len(x), n_classes) or not bool(
            torch.isfinite(logits).all()
        ):
            raise InputError(
                "Predictor must return finite [batch, declared classes] logits"
            )
        return logits.detach()
    finally:
        for module, flag in flags:
            module.training = flag


def _cost(logits, y, calibrator):
    r"""Compute a single categorical data-code term from normalized predictions.

    Args:
        logits (Tensor): Floating [batch, classes] logits.
        y (LongTensor): Labels of the batch that has just been predicted.
        calibrator (object, optional): Current causal log_probs component.

    Returns:
        Tensor: Detached scalar -sum_i log p(y_i), in nats.

    Raises:
        InputError: If calibration changes shape or returns unnormalized/nonfinite values.

    Notes:
        O(batch * classes) time and storage; each observation is charged once.
    """
    log_probs = (
        F.log_softmax(logits, dim=-1)
        if calibrator is None
        else calibrator.log_probs(logits)
    )
    if log_probs.shape != logits.shape or not bool(torch.isfinite(log_probs).all()):
        raise InputError(
            "Calibrator must return finite log probabilities with unchanged shape"
        )
    if not torch.allclose(
        torch.logsumexp(log_probs, dim=-1),
        torch.zeros_like(log_probs[:, 0]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise InputError("Calibrator probabilities must be normalized")
    return -log_probs.gather(1, y.to(logits.device)[:, None]).sum().detach()


def _initial_cost(initial_code, count, n_classes, x):
    r"""Use an explicit initial code or a uniform class-alphabet code.

    Args:
        initial_code (float or Tensor, optional): Total initial prefix cost in nats.
        count (int): Number of initially encoded observations.
        n_classes (int): Shared class alphabet.
        x (Tensor): Determines default scalar device and floating dtype.

    Returns:
        Tensor: Nonnegative scalar, default count * log(n_classes).

    Raises:
        InputError: If a supplied cost is invalid or nonempty for an empty prefix.

    Notes:
        O(1) time and storage, not a pass over prefix labels.
    """
    if initial_code is None:
        return torch.tensor(
            count * math.log(n_classes),
            dtype=x.dtype if x.is_floating_point() else torch.float64,
            device=x.device,
        )
    cost = scalar(initial_code, "initial_code")
    if bool(cost < 0) or (count == 0 and bool(cost != 0)):
        raise InputError(
            "Initial code must be nonnegative and empty when initial_count=0"
        )
    return cost


def prepare_blockwise(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    boundaries: Sequence[int],
    model_factory: Callable,
    train: Callable,
    n_classes: int,
    predict_logits: Callable | None = None,
    initial_code=None,
    calibrator_factory: Callable | None = None,
    provenance=None,
):
    r"""Chunk-incremental/from-scratch code, with prefix-only training.

    ``boundaries`` contains exclusive block ends, ending at n. The first block
    uses the declared initial code (uniform alphabet by default). Each later
    block uses a fresh ``model_factory()`` trained on exactly ``x[:start]`` and
    ``y[:start]``. ``train`` returns a replacement state or None for in-place use.

    Args:
        x (Tensor): Ordered observations [n, ...].
        y (LongTensor): Ordered labels [n].
        boundaries (Sequence[int]): Strictly increasing exclusive block ends, last n.
        model_factory (callable): Fresh learner state for every noninitial block.
        train (callable): (state, prefix_x, prefix_y) -> replacement state or None.
        n_classes (int): Fixed shared alphabet size.
        predict_logits (callable, optional): (state, x) -> logits [batch, classes].
        initial_code (float or Tensor, optional): Prefix cost; uniform by default.
        calibrator_factory (callable, optional): Fresh past-only calibrator per block.
        provenance (Provenance, optional): User-declared identities.

    Returns:
        PrequentialLedger: Detached categorical costs and causal interval metadata.

    Raises:
        InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

    Notes:
        Callbacks receive only the prefix/revealed slices, but their closed-over
        state cannot be audited. Input tensors are already materialized; this is
        not an out-of-core streaming data loader. Costs of user training callbacks
        must be added explicitly to any arithmetic complexity estimate.
        For block starts s_j and sizes b_j, work is sum_j Train(s_j)+F(b_j),
        plus optional calibration on each prefix. Persistent ledger storage is O(B);
        weak-reference bookkeeping does not retain all historical models.
    """
    n = _sequence(x, y, n_classes)
    boundaries = tuple(boundaries)
    if not boundaries or boundaries[-1] != n:
        raise InputError("Block boundaries must end at n")
    previous = 0
    for stop in boundaries:
        positive_int(stop, "block boundary")
        if stop <= previous or stop > n:
            raise InputError("Block boundaries must be strictly increasing")
        previous = stop
    initial_count = boundaries[0]
    initial = _initial_cost(initial_code, initial_count, n_classes, x)
    entries = []
    factory_references = []
    for start, stop in zip(boundaries, boundaries[1:]):
        state = model_factory()
        # Weak references catch an accidentally reused nn.Module without keeping
        # all prefix models alive. Non-weakrefable custom states remain a caller contract.
        factory_references = [
            reference for reference in factory_references if reference() is not None
        ]
        if any(reference() is state for reference in factory_references):
            raise InputError(
                "Chunk-from-scratch model_factory must return an independent model"
            )
        try:
            factory_references.append(weakref.ref(state))
        except TypeError:
            pass
        trained = train(state, x[:start], y[:start])
        if trained is not None:
            state = trained
        calibrator = None if calibrator_factory is None else calibrator_factory()
        calibrated_until = 0
        if calibrator is not None:
            past_logits = _logits(state, x[:start], predict_logits, n_classes)
            calibrator.update(past_logits, y[:start].to(past_logits.device))
            calibrated_until = start
        logits = _logits(state, x[start:stop], predict_logits, n_classes)
        entries.append(
            PrequentialEntry(
                start,
                stop,
                _cost(logits, y[start:stop], calibrator),
                start,
                calibrated_until,
            )
        )
    return PrequentialLedger(
        initial, initial_count, tuple(entries), n, "blockwise", provenance
    )


def prepare_online(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model_factory: Callable,
    update: Callable,
    n_classes: int,
    batch_size: int = 1,
    predict_logits: Callable | None = None,
    initial_count: int = 0,
    initial_code=None,
    calibrator=None,
    provenance=None,
):
    r"""Online next-minibatch code: score, calibrator update, learner update.

    The callback is ``update(state, revealed_x, revealed_y)``. Only the new
    scored minibatch is passed. Persistent learner/optimizer state is owned by
    the caller's state object; this function does not infer an optimizer.

    Args:
        x (Tensor): Materialized observations [n, ...].
        y (LongTensor): Labels [n].
        model_factory (callable): Creates the initial learner state.
        update (callable): (state, new_x, new_y) -> state or None, after scoring.
        n_classes (int): Fixed shared alphabet size.
        batch_size (int): Positive next-block size.
        predict_logits (callable, optional): Explicit state-to-logits adapter.
        initial_count (int): Initially encoded prefix length, possibly zero.
        initial_code (float or Tensor, optional): Prefix code in nats.
        calibrator (object, optional): Mutable past-only temperature component.
        provenance (Provenance, optional): User-declared identities.

    Returns:
        PrequentialLedger: Detached categorical costs and causal interval metadata.

    Raises:
        InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

    Notes:
        Callbacks receive only the prefix/revealed slices, but their closed-over
        state cannot be audited. Input tensors are already materialized; this is
        not an out-of-core streaming data loader. Costs of user training callbacks
        must be added explicitly to any arithmetic complexity estimate.
        Work is sum_j[F(b_j)+Update(b_j)+Calibration(b_j)], plus O(n*C)
        categorical reductions. Ledger storage is O(ceil(n/batch_size)).
    """
    return _prepare_stream(
        x,
        y,
        model_factory=model_factory,
        update=update,
        n_classes=n_classes,
        batch_size=batch_size,
        predict_logits=predict_logits,
        initial_count=initial_count,
        initial_code=initial_code,
        calibrator=calibrator,
        provenance=provenance,
        protocol="online",
    )


def prepare_replay(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model_factory: Callable,
    update: Callable,
    n_classes: int,
    replay_size: int,
    batch_size: int = 1,
    predict_logits: Callable | None = None,
    initial_count: int = 0,
    initial_code=None,
    calibrator=None,
    replay_policy: Callable | None = None,
    seed: int = 0,
    provenance=None,
):
    r"""Uniform-prefix replay code; replay losses are not encoded again.

    Callback: ``update(state, new_x, new_y, replay_x, replay_y)``. Replay is drawn
    after scoring the new minibatch, from indices below its exclusive stop.
    A custom ``replay_policy(revealed_count)`` may return only such indices.
    This is exact revealed-prefix random replay, not the paper's replay-streams
    memory optimization or a claim to reproduce its empirical experiments.

    Args:
        x (Tensor): Materialized observations [n, ...].
        y (LongTensor): Labels [n].
        model_factory (callable): Creates callback-owned learner state.
        update (callable): (state, new_x, new_y, replay_x, replay_y) -> state or None.
        n_classes (int): Declared alphabet size.
        replay_size (int): Maximum sampled revealed indices per update.
        batch_size (int): Positive number of newly scored observations.
        predict_logits (callable, optional): Explicit prediction adapter.
        initial_count (int): Uniform/explicit-code prefix length.
        initial_code (float or Tensor, optional): Initial prefix cost in nats.
        calibrator (object, optional): Updated only after scoring a batch.
        replay_policy (callable, optional): (revealed_count) -> allowed indices.
        seed (int): Seed for the local replay generator.
        provenance (Provenance, optional): User-declared identities.

    Returns:
        PrequentialLedger: Detached categorical costs and causal interval metadata.

    Raises:
        InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

    Notes:
        Callbacks receive only the prefix/revealed slices, but their closed-over
        state cannot be audited. Input tensors are already materialized; this is
        not an out-of-core streaming data loader. Costs of user training callbacks
        must be added explicitly to any arithmetic complexity estimate.
        Default randperm(stop) costs O(stop) per update, even for a tiny replay_size:
        O(sum_j stop_j), up to O(n^2/batch_size). Retained replay indices cost
        O(B*min(replay_size,n)); selected data batches allocate gather copies.
    """
    if (
        isinstance(replay_size, bool)
        or not isinstance(replay_size, int)
        or replay_size < 0
    ):
        raise InputError("replay_size must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    return _prepare_stream(
        x,
        y,
        model_factory=model_factory,
        update=update,
        n_classes=n_classes,
        batch_size=batch_size,
        predict_logits=predict_logits,
        initial_count=initial_count,
        initial_code=initial_code,
        calibrator=calibrator,
        provenance=provenance,
        protocol="replay",
        replay_size=replay_size,
        replay_policy=replay_policy,
        generator=generator,
    )


def _prepare_stream(
    x,
    y,
    *,
    model_factory,
    update,
    n_classes,
    batch_size,
    predict_logits,
    initial_count,
    initial_code,
    calibrator,
    provenance,
    protocol,
    replay_size=0,
    replay_policy=None,
    generator=None,
):
    r"""Implement the shared score -> record -> calibrate -> update stream loop.

    Args:
        x (Tensor): Materialized observation sequence.
        y (LongTensor): Matching categorical labels.
        model_factory (callable): Initial state factory.
        update (callable): Protocol-specific update callback.
        n_classes (int): Shared categorical alphabet size.
        batch_size (int): Newly revealed block size.
        predict_logits (callable, optional): Prediction adapter.
        initial_count (int): Initial coded prefix length.
        initial_code (float or Tensor, optional): Initial cost in nats.
        calibrator (object, optional): Past-only mutable calibration state.
        provenance (Provenance, optional): User-declared identities.
        protocol (str): ``online`` or ``replay``.
        replay_size (int): Maximum default replay sample size.
        replay_policy (callable, optional): Explicit revealed-prefix index policy.
        generator (torch.Generator, optional): Local replay random generator.

    Returns:
        PrequentialLedger: Costs are fixed before labels enter an update callback.

    Notes:
        No complete optimizer history is retained. Apart from callback work, the
        online path costs O(n*C); default replay also materializes randperm(stop)
        on each step. The full input sequence and O(B+R_total) ledger stay alive.
    """
    n = _sequence(x, y, n_classes)
    positive_int(batch_size, "batch_size")
    if (
        isinstance(initial_count, bool)
        or not isinstance(initial_count, int)
        or not 0 <= initial_count <= n
    ):
        raise InputError("initial_count must lie between zero and n")
    initial = _initial_cost(initial_code, initial_count, n_classes, x)
    state = model_factory()
    if initial_count:
        if protocol == "online":
            replacement = update(state, x[:initial_count], y[:initial_count])
        else:
            # Initial labels have been encoded; replay may now use them too.
            replacement = update(
                state, x[:initial_count], y[:initial_count], x[:0], y[:0]
            )
        if replacement is not None:
            state = replacement
    entries = []
    calibrated_until = 0
    for start in range(initial_count, n, batch_size):
        stop = min(start + batch_size, n)
        logits = _logits(state, x[start:stop], predict_logits, n_classes)
        nll = _cost(logits, y[start:stop], calibrator)
        replay_indices = ()
        if protocol == "replay":
            indices = (
                torch.randperm(stop, generator=generator)[
                    : min(replay_size, stop)
                ].tolist()
                if replay_policy is None
                else list(replay_policy(stop))
            )
            if any(
                isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < stop
                for i in indices
            ):
                raise InputError("Replay may access only revealed indices below stop")
            replay_indices = tuple(indices)
        entries.append(
            PrequentialEntry(start, stop, nll, start, calibrated_until, replay_indices)
        )
        # Everything below occurs AFTER the corresponding data code was fixed.
        if calibrator is not None:
            calibrator.update(logits, y[start:stop].to(logits.device))
            calibrated_until = stop
        if protocol == "online":
            replacement = update(state, x[start:stop], y[start:stop])
        else:
            replay_x = x[
                torch.tensor(replay_indices, device=x.device, dtype=torch.long)
            ]
            replay_y = y[
                torch.tensor(replay_indices, device=y.device, dtype=torch.long)
            ]
            replacement = update(
                state, x[start:stop], y[start:stop], replay_x, replay_y
            )
        if replacement is not None:
            state = replacement
    return PrequentialLedger(
        initial, initial_count, tuple(entries), n, protocol, provenance
    )
