"""Closed-form norms and derivatives through the public PyTorch integrations.

Weight norms are diagnostics, not reparameterization-invariant statistical
complexity. The expected failure makes an existing low-precision count limitation
visible instead of treating an inaccurately rounded integer as an exact oracle.
"""

import math

import pytest
import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.criteria import BIC
from dl_complexity.diagnostics import WeightInputs, WeightStatistics
from dl_complexity.integrations.torch import GaussianLikelihood


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_diagonal_linear_checkpoint_norms(dtype, tmp_path, record_analytic):
    """Recover known singular values and norms from an ordinary state_dict."""
    model = nn.Linear(3, 3, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.diag(torch.tensor([3.0, -2.0, 1.0], dtype=dtype)))
    path = tmp_path / "diagonal.pt"
    torch.save(model.state_dict(), path)
    context = ModelContext.from_checkpoint(
        path, model_factory=lambda: nn.Linear(3, 3, bias=False, dtype=dtype)
    )
    result = WeightStatistics(spectral=True).estimate(context)
    assert result.primary is None
    expected = {
        "parameter_count": (9.0, "count", "3*3 dense stored coefficients"),
        "l1": (6.0, "weight_norm", "sum(abs(diag(3,-2,1)))=6"),
        "l2": (math.sqrt(14), "weight_norm", "sqrt(3^2+(-2)^2+1^2)"),
        "weight.l2": (math.sqrt(14), "weight_norm", "Frobenius norm of diagonal W"),
        "weight.spectral": (3.0, "weight_norm", "max absolute diagonal entry"),
        "sparsity": (2 / 3, "dimensionless", "six exact zeros / nine coefficients"),
    }
    for name, (value, unit, oracle) in expected.items():
        quantity = result.quantities[name]
        assert quantity.unit == unit
        record_analytic(
            case_id=f"diagonal_checkpoint.{dtype}.{name}",
            method="WeightStatistics",
            model="Linear diagonal W=diag(3,-2,1), checkpoint round-trip",
            expected=value,
            actual=quantity.value,
            unit=unit,
            oracle=oracle,
            atol=1e-12 if dtype == torch.float64 else 3e-7,
            rtol=1e-12 if dtype == torch.float64 else 1e-7,
            details={"dtype": str(dtype)},
        )


def test_equivalent_linear_parameterizations_have_different_norms(record_analytic):
    """A fixed function does not determine a unique weight-norm complexity."""
    inputs = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], dtype=torch.float64)
    for scale in (1.0, 4.0):
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False, dtype=torch.float64),
            nn.Linear(2, 2, bias=False, dtype=torch.float64),
        )
        with torch.no_grad():
            model[0].weight.copy_(scale * torch.eye(2, dtype=torch.float64))
            model[1].weight.copy_(torch.eye(2, dtype=torch.float64) / scale)
        assert torch.equal(model(inputs), inputs)
        result = WeightStatistics().estimate(ModelContext(model))
        record_analytic(
            case_id=f"identity_reparameterization.scale{scale}.l2",
            method="WeightStatistics",
            model="Two Linear layers: (c*I, I/c), same identity predictor",
            expected=math.sqrt(2 * scale**2 + 2 / scale**2),
            actual=result.quantities["l2"].value,
            unit="weight_norm",
            oracle="||[c*I,I/c]||_2 = sqrt(2*c^2+2/c^2)",
            details={"scale": scale, "same_predictor": True},
        )


def test_gaussian_score_gradient_and_hessian(record_analytic):
    """Differentiate a fixed-weight score; no MLE applicability is asserted here."""
    xs = [(1.0, 0.0), (1.0, 2.0), (-1.0, 1.0)]
    ys = [2.0, -1.0, 0.5]
    weights = [0.5, -1.5]
    sigma = 1.7
    n, k = len(xs), len(weights)
    residuals = [
        y - math.fsum(xj * wj for xj, wj in zip(x, weights)) for x, y in zip(xs, ys)
    ]
    expected_ll = -n * math.log(sigma * math.sqrt(2 * math.pi))
    expected_ll -= math.fsum(r * r for r in residuals) / (2 * sigma**2)
    model = nn.Linear(k, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([weights], dtype=torch.float64))
    context = ModelContext(model, likelihood=GaussianLikelihood(sigma=sigma))
    data = DataSource.from_tensors(
        torch.tensor(xs, dtype=torch.float64),
        torch.tensor(ys, dtype=torch.float64)[:, None],
    )
    result = BIC().estimate(context, data=data)
    gradient = torch.autograd.grad(
        result.primary.value, model.weight, create_graph=True
    )[0]
    record_analytic(
        case_id="gaussian_fixed_weight.bic_value",
        method="BIC",
        model="Gaussian Linear at explicitly fixed, non-MLE weights",
        expected=-2 * expected_ll + k * math.log(n),
        actual=result.primary.value,
        unit="native_ic",
        oracle="Score arithmetic -2*ell(w)+k*ln(n); not an MLE validity claim",
    )
    for j in range(k):
        expected_grad = (
            -2 * math.fsum(x[j] * r for x, r in zip(xs, residuals)) / sigma**2
        )
        record_analytic(
            case_id=f"gaussian_fixed_weight.gradient{j}",
            method="BIC/autograd",
            model="Gaussian Linear with known variance",
            expected=expected_grad,
            actual=gradient[0, j],
            unit="native_ic/weight",
            oracle="d[-2*ell(w)]/dw_j = -2*sum_i X_ij*(y_i-X_i*w)/sigma^2",
        )
        hessian_row = torch.autograd.grad(
            gradient[0, j], model.weight, retain_graph=True
        )[0]
        for h in range(k):
            record_analytic(
                case_id=f"gaussian_fixed_weight.hessian{j}{h}",
                method="BIC/autograd",
                model="Gaussian Linear with known variance",
                expected=2 * math.fsum(x[j] * x[h] for x in xs) / sigma**2,
                actual=hessian_row[0, h],
                unit="native_ic/weight^2",
                oracle="Hessian[-2*ell(w)] = 2*X.T*X/sigma^2",
            )


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: parameter_count uses weight dtype; float16 rounds 2049",
)
def test_parameter_count_should_not_round_integer_dimension(record_analytic):
    """Expose, rather than hide, the currently documented float16 count issue."""
    weights = torch.ones(2049, dtype=torch.float16)
    result = WeightStatistics().estimate(inputs=WeightInputs({"weight": weights}))
    record_analytic(
        case_id="known_limitation.float16_parameter_count_2049",
        method="WeightStatistics",
        model="A tensor with exactly 2049 scalar weights",
        expected=2049,
        actual=result.quantities["parameter_count"].value,
        unit="count",
        oracle="Integer tensor.numel() is exactly 2049; counts should be exact",
        atol=0.0,
        rtol=0.0,
        details={"known_limitation": True, "dtype": "torch.float16"},
    )
