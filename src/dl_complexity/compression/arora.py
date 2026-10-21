"""Arora et al. 2018 Matrix-Project with a finite coefficient alphabet.

Implements the matrix projection in Algorithm 1 (paper physical p. 6), not the
full Theorem 4.1 compression-budget/noise-stability construction. Helpers and
the coefficient alphabet must be fixed independently of training observations
to use the resulting predictor family with Theorem 2.1.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .base import ModelTransform
from .codecs import _architecture, _build_architecture


def _arora_schema(model: nn.Module) -> str:
    r"""Validate the entire predictor and bind a canonical, clean decoder schema.

    Checking named_parameters alone is insufficient: buffers, hooks or an
    instance-level forward can hide arbitrary additional learned predictors.
    This narrow language excludes all such behavior rather than charging an
    unknown amount of side information to the finite coefficient budget.

    Args:
        model (nn.Module): Bias-free built-in Linear/ReLU/Sequential predictor.

    Returns:
        str: Canonical JSON architecture with deterministic eval-mode flags.

    Raises:
        ValueError: If custom methods, hooks, buffers, aliases or extra state are present.

    Notes:
        Model metadata/schema traversal only; O(L^2 + H) worst-case work from the
        recursive architecture describer and O(H + number of parameters) metadata.
        This binds the decoder language, not statistical independence from data.
    """
    module_runtime = torch.nn.modules.module
    if module_runtime._global_forward_hooks or module_runtime._global_forward_pre_hooks:
        raise ValueError(
            "global forward hooks are not part of the Arora decoder schema"
        )
    supported = (nn.Sequential, nn.Linear, nn.ReLU, nn.Identity, nn.Flatten)

    def validate(module: nn.Module) -> None:
        r"""Recursively reject predictor state not represented by the decoder language.

        Args:
            module (nn.Module): Current subtree checked by exact built-in class identity.

        Raises:
            ValueError: If behavior, buffers, parameters or child modules are unsupported.

        Notes:
            O(L + metadata fields) validation work; no forward or tensor-value scan.
        """
        if type(module) not in supported:
            raise ValueError(
                "Arora decoder supports only bias-free built-in Linear/ReLU MLPs"
            )
        for name, value in module.__dict__.items():
            if callable(value) and (
                hasattr(type(module), name) or name == "_compiled_call_impl"
            ):
                raise ValueError(
                    "instance forward/call or other method overrides are not in the Arora schema"
                )
        if any(
            getattr(module, name, {})
            for name in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ):
            raise ValueError("module hooks are not part of the Arora decoder schema")
        if module._buffers:
            raise ValueError(
                "buffers, including nonpersistent buffers, are not in the Arora decoder schema"
            )
        if type(module) is nn.Linear:
            if set(module._parameters) != {"weight", "bias"} or module.bias is not None:
                raise ValueError(
                    "only the declared bias-free Linear weight belongs to the Arora schema"
                )
            if (
                not isinstance(module.weight, nn.Parameter)
                or not module.weight.is_floating_point()
                or tuple(module.weight.shape)
                != (module.out_features, module.in_features)
            ):
                raise ValueError(
                    "Linear parameter disagrees with the Arora architecture schema"
                )
        elif module._parameters:
            raise ValueError(
                "unexpected parameters are not part of the Arora decoder schema"
            )
        if type(module) is not nn.Sequential and module._modules:
            raise ValueError(
                "unexpected child layers are not part of the Arora decoder schema"
            )
        for child in module._modules.values():
            validate(child)

    validate(model)
    parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(parameter) for _, parameter in parameters}) != len(parameters):
        raise ValueError(
            "tied parameters need a separate finite-parameterization contract"
        )
    storages = [
        (str(p.device), p.untyped_storage().data_ptr())
        for _, p in parameters
        if p.numel()
    ]
    if len(set(storages)) != len(storages):
        raise ValueError(
            "shared parameter storage is not part of the Arora decoder schema"
        )
    schema = _architecture(model)

    def fixed_eval(description: dict) -> None:
        # All supported operations have the same prediction in train/eval mode.
        # Mode changes are harmless, unlike changes of architecture or hooks.
        r"""Canonicalize harmless training flags in a recursively described schema.

        Args:
            description (dict): Mutable local architecture description.

        Notes:
            Mutates only the new metadata dictionary; O(L) work and O(depth) recursion.
        """
        description["training"] = False
        if description["type"] == "Sequential":
            for _, child in description["options"]["children"]:
                fixed_eval(child)

    fixed_eval(schema)
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validated_reconstruction(projection, parameter: Tensor) -> Tensor:
    r"""Validate finite projection artifacts against one reference parameter.

    Args:
        projection (MatrixProjection): Fixed helpers, alphabet and learned finite indices.
        parameter (Tensor): Reference shape/dtype/device for this parameter.

    Returns:
        Tensor: Reconstructed finite predictor matrix.

    Raises:
        ValueError: If helpers, alphabet, indices or reconstruction are inconsistent.

    Notes:
        For K helpers and P matrix entries, O(K*P + r) work/storage is dominated
        by helper checks and reconstruction. Accelerator boolean checks synchronize.
    """
    if not isinstance(projection, MatrixProjection):
        raise ValueError("finite representation must contain MatrixProjection objects")
    helper, alphabet, indices = (
        projection.helper,
        projection.coefficient_alphabet,
        projection.coefficient_indices,
    )
    if (
        not isinstance(helper, Tensor)
        or helper.ndim != 3
        or helper.shape[0] == 0
        or tuple(helper.shape[1:]) != tuple(parameter.shape)
        or helper.dtype != parameter.dtype
        or helper.device != parameter.device
        or not bool(((helper == 1) | (helper == -1)).all())
    ):
        raise ValueError("invalid fixed Rademacher helper in finite representation")
    if (
        not isinstance(alphabet, Tensor)
        or alphabet.ndim != 1
        or alphabet.numel() == 0
        or alphabet.dtype != parameter.dtype
        or alphabet.device != parameter.device
        or not torch.isfinite(alphabet).all()
        or not bool((alphabet[1:] > alphabet[:-1]).all())
    ):
        raise ValueError("invalid finite coefficient alphabet")
    if (
        not isinstance(indices, Tensor)
        or indices.ndim != 1
        or indices.numel() != helper.shape[0]
        or indices.dtype != torch.int64
        or indices.device != parameter.device
        or not bool(((indices >= 0) & (indices < alphabet.numel())).all())
    ):
        raise ValueError("invalid finite coefficient indices")
    reconstructed = projection.reconstructed
    if not torch.isfinite(reconstructed).all():
        raise ValueError("decoded finite predictor overflowed")
    return reconstructed


@dataclass(frozen=True)
class MatrixProjection:
    r"""A matrix represented by finite coefficients in a fixed helper basis.

    Args:
        coefficient_indices (LongTensor): K indices into the fixed alphabet.
        coefficient_alphabet (Tensor): Strictly ordered finite grid [r].
        helper (Tensor): Fixed Rademacher matrices [K, rows, cols].
        max_coefficient_rounding_error (Tensor): Largest scalar quantization error.

    Notes:
        Record creation is O(1). Referenced storage is O(K*P + r + K); freezing the
        dataclass does not freeze Tensor contents. Reconstruction is computed afresh.
    """

    coefficient_indices: Tensor
    coefficient_alphabet: Tensor
    helper: Tensor
    max_coefficient_rounding_error: Tensor

    @property
    def reconstructed(self) -> Tensor:
        r"""Decode finite indices in the fixed helper basis, without fitted state.

        Returns:
            Tensor: mean_k alphabet[index_k] * helper_k, with shape [rows, cols].

        Notes:
            Every property access performs O(K*P) multiplication/reduction and allocates
            a broadcast [K, rows, cols] temporary; this is not an O(1) cached attribute.
        """
        coefficients = self.coefficient_alphabet[self.coefficient_indices]
        return (coefficients[:, None, None] * self.helper).mean(dim=0)

    @property
    def parameter_count(self) -> int:
        r"""Return K, the number of finite coefficient coordinates, in O(1).

        Returns:
            int: Number of stored coefficient indices, not number of dense weights.
        """
        return self.coefficient_indices.numel()

    @property
    def values_per_parameter(self) -> int:
        r"""Return fixed alphabet size r in O(1).

        Returns:
            int: Maximum number of values for each finite coefficient coordinate.
        """
        return self.coefficient_alphabet.numel()


@dataclass(frozen=True)
class ProjectedModel:
    r"""Independent projected model and the actual finite representation sizes.

    Args:
        model (nn.Module): Independent convenience predictor matching finite indices.
        projections (Mapping[str, MatrixProjection]): Complete parameter-name mapping.
        architecture_json (str): Immutable canonical architecture binding.

    Notes:
        The record owns references, not deep immutability of tensors or model. decode
        revalidates all mutable state and creates a clean predictor before certification.
    """

    model: nn.Module
    projections: Mapping[str, MatrixProjection]
    architecture_json: str

    def decode(self) -> nn.Module:
        r"""Rebuild g from the bound schema and finite indices, not mutable forward.

        The convenience model is revalidated and must still match both schema
        and code. No user-provided instance, hook, extra buffer or overridden
        forward is used to produce the certified compressed logits.

        Returns:
            nn.Module: Fresh predictor built only from canonical schema and finite indices.

        Raises:
            ValueError: If schema, aliases, dtype/device or convenience weights changed.

        Notes:
            O(sum_l K_l*P_l + schema work) time and similarly sized broadcast temporaries,
            plus O(P) decoded state copies. Mutable instance forwards are never trusted.
        """
        if _arora_schema(self.model) != self.architecture_json:
            raise ValueError(
                "projected predictor architecture no longer matches its bound schema"
            )
        parameters = dict(self.model.named_parameters())
        if set(parameters) != set(self.projections) or not parameters:
            raise ValueError(
                "finite representation must cover every predictor parameter"
            )
        decoded_state = {}
        for name, parameter in parameters.items():
            reconstruction = _validated_reconstruction(
                self.projections[name], parameter
            )
            if not torch.equal(parameter.detach(), reconstruction):
                raise ValueError(
                    f"projected parameter {name} no longer matches its finite code"
                )
            decoded_state[name] = reconstruction.detach().clone()
        decoded = _build_architecture(json.loads(self.architecture_json))
        decoded.load_state_dict(decoded_state, strict=True, assign=True)
        return decoded

    @property
    def parameter_count(self) -> int:
        r"""Sum all finite coefficient coordinates in O(number of weight matrices).

        Returns:
            int: q for the declared finite predictor family.
        """
        return sum(
            projection.parameter_count for projection in self.projections.values()
        )

    @property
    def values_per_parameter(self) -> int:
        r"""Find the largest fixed coefficient alphabet in O(number of matrices).

        Returns:
            int: r such that product_l r_l**K_l <= r**q.
        """
        return max(
            projection.values_per_parameter for projection in self.projections.values()
        )


def matrix_project(
    matrix: Tensor, helper: Tensor, coefficient_alphabet: Tensor
) -> MatrixProjection:
    r"""Compute sum_k <A,M_k> M_k / K, after declared finite-grid rounding.

    ``helper`` has shape [K, rows, cols] with fixed Rademacher entries +/-1.
    ``coefficient_alphabet`` is a finite, sorted, data-independent grid. Nearest
    grid value is selected; exact ties select its lower index. Quantization may
    increase error, which MUST be checked on all training logits afterward.
    No probability certificate is fabricated from a requested projection size.

    Args:
        matrix (Tensor): Finite floating matrix [rows, cols].
        helper (Tensor): Fixed +/-1 matrices [K, rows, cols], same dtype/device.
        coefficient_alphabet (Tensor): Fixed strictly increasing grid [r].

    Returns:
        MatrixProjection: Finite indices, copied helper/grid and rounding diagnostic.

    Raises:
        ValueError: If values, shapes, dtype/device or coefficient arithmetic are invalid.

    Notes:
        Unquantized z_k = <A,M_k>; each z_k is replaced by its nearest grid point.
        Coefficients cost O(K*P), nearest-grid search O(K*r), with corresponding
        broadcast temporaries. Returned helper clones cost O(K*P); all work is no_grad.
    """
    if (
        not isinstance(matrix, Tensor)
        or matrix.ndim != 2
        or not matrix.is_floating_point()
        or not torch.isfinite(matrix).all()
    ):
        raise ValueError("matrix_project requires a finite floating matrix")
    if (
        not isinstance(helper, Tensor)
        or helper.ndim != 3
        or helper.shape[0] < 1
        or tuple(helper.shape[1:]) != tuple(matrix.shape)
        or helper.device != matrix.device
        or helper.dtype != matrix.dtype
    ):
        raise ValueError("helper must have matching [K, rows, cols], dtype and device")
    if not bool(((helper == 1) | (helper == -1)).all()):
        raise ValueError("Algorithm 1 requires Rademacher helper entries")
    alphabet = coefficient_alphabet
    if (
        not isinstance(alphabet, Tensor)
        or alphabet.ndim != 1
        or alphabet.numel() == 0
        or alphabet.dtype != matrix.dtype
        or alphabet.device != matrix.device
        or not torch.isfinite(alphabet).all()
        or not bool((alphabet[1:] > alphabet[:-1]).all())
    ):
        raise ValueError(
            "coefficient alphabet must be finite, strictly increasing and match dtype/device"
        )
    with torch.no_grad():
        coefficients = (helper * matrix.unsqueeze(0)).sum(dim=(1, 2))
        if not torch.isfinite(coefficients).all():
            raise ValueError("Matrix-Project coefficient calculation overflowed")
        indices = (coefficients[:, None] - alphabet[None, :]).abs().argmin(dim=1)
        rounded = alphabet[indices]
        rounding_error = (rounded - coefficients).abs().max()
    return MatrixProjection(
        indices, alphabet.detach().clone(), helper.detach().clone(), rounding_error
    )


def make_projection_helpers(
    model: nn.Module, n_projections: int, *, seed: int
) -> dict[str, Tensor]:
    r"""Generate fixed Rademacher helpers using a local CPU generator.

    Only architecture shapes/dtypes/devices are read, not learned weights.
    Calling this after training does not establish independence by itself: the
    user must have fixed seed, architecture and projection count before data.

    Args:
        model (nn.Module): Supported bias-free model whose architecture is already fixed.
        n_projections (int): Positive K helper matrices per weight matrix.
        seed (int): Independently predeclared local CPU generator seed.

    Returns:
        dict[str, Tensor]: Parameter-name -> [K, rows, cols] Rademacher helpers.

    Raises:
        ValueError: If model language or configuration is unsupported.

    Notes:
        O(K*P) generation/storage, including CPU integer tensors and device copies.
        It does not consume global RNG or read weight values; late seed selection
        still cannot establish independence of the helper string from the sample.
    """
    if (
        isinstance(n_projections, bool)
        or not isinstance(n_projections, int)
        or n_projections < 1
    ):
        raise ValueError("n_projections must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _arora_schema(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not parameter.is_floating_point():
            raise ValueError(
                "Matrix-Project model helpers support bias-free matrix parameters only"
            )
        signs = torch.randint(
            0, 2, (n_projections, *parameter.shape), generator=generator
        )
        result[name] = (2 * signs - 1).to(
            device=parameter.device, dtype=parameter.dtype
        )
    if not result:
        raise ValueError("no weight matrices to project")
    return result


def nominal_projection_count(epsilon: float, eta: float) -> int:
    r"""Paper Algorithm 1 budget ceil(log(1/eta)/epsilon**2).

    This is the algorithm's nominal budget, not an independently verified
    finite-sample failure-probability guarantee for quantized coefficients.
    Theorem 2.1 still needs a measured sample-compressibility certificate.

    Args:
        epsilon (float): Nominal error parameter in (0, 1].
        eta (float): Nominal failure probability in (0, 1).

    Returns:
        int: ceil(-log(eta) / epsilon**2).

    Raises:
        ValueError: If error/failure parameters are outside their declared range.

    Notes:
        O(1) arithmetic. This is not a validated allocation of a complete network's
        geometry-dependent Theorem 4.1 budget or a guarantee after grid rounding.
    """
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    if not math.isfinite(eta) or not 0 < eta < 1:
        raise ValueError("eta must be in (0, 1)")
    return math.ceil(-math.log(eta) / epsilon**2)


class AroraMatrixProject(ModelTransform):
    """Project all weights of a bias-free small Linear/ReLU MLP.

    Constructor stores only fixed helper/alphabet configuration. ``project``
    returns learned finite indices and an independent model; no fit artifact
    is cached inside this object. Unlike data-derived min/max quantization, a
    fixed alphabet gives a declared r**q family conditional on fixed helpers.
    """

    def __init__(self, helpers: Mapping[str, Tensor], coefficient_alphabet: Tensor):
        r"""Store fixed projection configuration, without learning coefficients.

        Args:
            helpers (Mapping[str, Tensor]): Fixed helper bases by parameter name.
            coefficient_alphabet (Tensor): Fixed finite coefficient grid shared by matrices.

        Notes:
            Copies the mapping in O(number of matrices), not its tensor storage. Keep
            helpers/grid unchanged and independent of the training sample.
        """
        self.helpers = dict(helpers)
        self.coefficient_alphabet = coefficient_alphabet

    def project(self, model: nn.Module) -> ProjectedModel:
        r"""Project every weight matrix and bind a clean finite-decoder architecture.

        Args:
            model (nn.Module): Bias-free supported model matching the configured helpers.

        Returns:
            ProjectedModel: Convenience model, finite indices and canonical schema.

        Raises:
            ValueError: If model/helper names, state or numerical values are unsupported.

        Notes:
            O(P) model copy plus sum_l O(K_l*P_l + K_l*r) projection work. The result
            retains helpers and indices; it is not automatically a smaller dense model.
        """
        architecture_json = _arora_schema(model)
        parameters = dict(model.named_parameters())
        if set(parameters) != set(self.helpers):
            raise ValueError("helper names must match every model parameter")
        result = deepcopy(model)
        projections = {}
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                projection = matrix_project(
                    parameter, self.helpers[name], self.coefficient_alphabet
                )
                parameter.copy_(projection.reconstructed)
                projections[name] = projection
        return ProjectedModel(result, projections, architecture_json)

    def apply(self, model: nn.Module) -> nn.Module:
        r"""Return only the convenience projected model from the richer project result.

        Args:
            model (nn.Module): Supported source model.

        Returns:
            nn.Module: Independent projected predictor.

        Notes:
            Has the same projection cost as project. Use project instead when a later
            Arora certificate must derive q/r from actual finite indices.
        """
        return self.project(model).model
