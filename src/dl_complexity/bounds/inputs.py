"""Prepared inputs for the finite-parameter Arora compression theorem."""

from dataclasses import dataclass

from torch import Tensor

from ..results import Provenance


@dataclass(frozen=True)
class AroraCompressionInputs:
    r"""A compression certificate and a fixed finite parameterization of g.

    ``parameter_count`` is a predeclared upper budget q, not necessarily the
    number of nonzero weights. Each of those q coordinates has at most r fixed
    discrete values. Learned scales/codebooks/support indices must be included
    in that parameterization; a serialized byte count is NOT a replacement.
    The alphabet, decoder and helper string must be fixed independently of S.

    ``max_logit_error`` bounds the coordinatewise error on the ENTIRE training
    sample. For margin threshold gamma we require error <= gamma/2, so the
    multiclass margin cannot cross zero outside the gamma-margin error set.
    Inputs can be explicitly user supplied or prepared from measured logits;
    neither a boolean declaration nor this dataclass proves independence.

    Args:
        empirical_margin_loss (Tensor or float): Fraction of sample margins <= gamma.
        n_observations (int): Full training sample size n.
        parameter_count (int): Predeclared finite coordinate budget q, not dense weights.
        values_per_parameter (int): Maximum finite coordinate alphabet size r.
        margin (float): Positive gamma threshold.
        max_logit_error (Tensor or float): Full-sample coordinate distortion <= gamma/2.
        helper_is_data_independent (bool): User declaration, not a measured fact.
        alphabet_is_data_independent (bool): User declaration for alphabet/q/r budget.
        certificate_kind (str): Origin label for measured or supplied summary.
        compressed_state_id (str, optional): Identity of the guaranteed predictor g.
        provenance (Provenance, optional): User-declared data/state identities.

    Notes:
        O(1) construction/storage apart from referenced scalar tensors. This record
        cannot infer a finite predictor family from arbitrary serialized byte length.
    """

    empirical_margin_loss: Tensor | float
    n_observations: int
    parameter_count: int
    values_per_parameter: int
    margin: float
    max_logit_error: Tensor | float
    helper_is_data_independent: bool
    alphabet_is_data_independent: bool
    certificate_kind: str = "user_supplied_summary"
    compressed_state_id: str | None = None
    provenance: Provenance | None = None
