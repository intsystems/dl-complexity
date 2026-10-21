"""Known-number and certificate oracles for the Arora compression bound."""

import math

import pytest
import torch
from torch import nn

from dl_complexity import DataSource
from dl_complexity.bounds import (
    AroraCompressionBound,
    AroraCompressionInputs,
    arora_inputs_from_logits,
    prepare_arora_compression,
    prepare_arora_projection,
)
from dl_complexity.compression import AroraMatrixProject
from dl_complexity.errors import InputError


def inputs(**changes):
    values = dict(
        empirical_margin_loss=0.1,
        n_observations=100,
        parameter_count=2,
        values_per_parameter=2,
        margin=1.0,
        max_logit_error=0.25,
        helper_is_data_independent=True,
        alphabet_is_data_independent=True,
    )
    values.update(changes)
    return AroraCompressionInputs(**values)


def test_explicit_arora_supplement_proof_oracle():
    result = AroraCompressionBound(delta=0.05).estimate(inputs=inputs())
    expected_penalty = math.sqrt((2 * math.log(2) + math.log(20)) / 200)
    assert result.primary.unit == "probability"
    assert result.primary.value.item() == pytest.approx(0.1 + expected_penalty)
    assert result.quantities["complexity_penalty"].value.item() == pytest.approx(
        0.14802071873007983
    )
    assert result.kind == "compressed_predictor_risk_bound"
    assert "Theorem2.1" in result.config["theorem"]
    assert any("NOT original" in value for value in result.assumptions)
    with pytest.raises(InputError):
        result.primary.to("bits")


def test_factor_two_counterexample_and_safe_certificate():
    original = torch.tensor([[1.5, 0.0]], dtype=torch.float64)
    compressed = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    assert original.argmax(1).item() == 0 and compressed.argmax(1).item() == 1
    # Error<=gamma is not enough for multiclass margins; error<=gamma/2 is.
    with pytest.raises(InputError, match="margin/2"):
        arora_inputs_from_logits(
            original,
            compressed,
            torch.tensor([0]),
            parameter_count=2,
            values_per_parameter=3,
            margin=1.0,
            helper_is_data_independent=True,
            alphabet_is_data_independent=True,
        )


def test_margin_tie_and_vacuous_bound_are_not_hidden():
    certificate = arora_inputs_from_logits(
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        torch.tensor([0]),
        parameter_count=2,
        values_per_parameter=3,
        margin=1.0,
        helper_is_data_independent=True,
        alphabet_is_data_independent=True,
    )
    assert certificate.empirical_margin_loss.item() == 1
    result = AroraCompressionBound().estimate(inputs=certificate)
    assert result.primary.value.item() > 1  # raw risk bound, no silent clip


def test_missing_independence_is_not_claimed_verified():
    method = AroraCompressionBound()
    provided = inputs(helper_is_data_independent=False)
    assert not method.check(inputs=provided).ready
    with pytest.raises(InputError):
        method.estimate(inputs=provided)
    with pytest.raises(InputError):
        method.estimate(inputs=inputs(parameter_count=-1))
    with pytest.raises(InputError):
        AroraCompressionBound(delta=1.0)
    result = AroraCompressionBound(delta=1e-320).estimate(inputs=inputs())
    assert torch.isfinite(result.primary.value)  # use -log(delta), not log(1/delta)


def test_full_sample_model_certificate_and_known_linear_value():
    original = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    compressed = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        original.weight.copy_(torch.tensor([[1.0], [-1.0]], dtype=torch.float64))
        compressed.weight.copy_(torch.tensor([[0.875], [-0.875]], dtype=torch.float64))
    x = torch.tensor([[1.0], [-1.0]], dtype=torch.float64).repeat(50, 1)
    y = torch.tensor([0, 1]).repeat(50)
    data = DataSource.from_tensors(x, y, data_id="oracle")
    rng = torch.random.get_rng_state().clone()
    certificate = prepare_arora_compression(
        original,
        compressed,
        data,
        parameter_count=2,
        values_per_parameter=3,
        margin=1.0,
        helper_is_data_independent=True,
        alphabet_is_data_independent=True,
        compressed_state_id="g",
    )
    assert original.training and compressed.training
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert certificate.max_logit_error.item() == 0.125
    assert certificate.empirical_margin_loss.item() == 0
    assert (
        certificate.provenance.state_id == "g"
        and certificate.provenance.data_id == "oracle"
    )
    result = AroraCompressionBound().estimate(inputs=certificate)
    assert result.primary.value.item() == pytest.approx(0.16113591857326862)


def test_real_matrix_project_to_arora_certificate_end_to_end():
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
    # Four orthogonal sign matrices span the four-dimensional weight space.
    helper = torch.tensor(
        [
            [[1, 1], [1, 1]],
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]],
        ],
        dtype=torch.float64,
    )
    projected = AroraMatrixProject(
        {"weight": helper}, torch.arange(-2, 3, dtype=torch.float64)
    ).project(model)
    torch.testing.assert_close(projected.model.weight, model.weight, rtol=0, atol=0)
    data = DataSource.from_tensors(
        torch.eye(2, dtype=torch.float64).repeat(50, 1), torch.tensor([0, 1]).repeat(50)
    )
    certificate = prepare_arora_projection(
        model,
        projected,
        data,
        margin=0.5,
        helper_is_data_independent=True,
        alphabet_is_data_independent=True,
    )
    result = AroraCompressionBound().estimate(inputs=certificate)
    expected = math.sqrt((4 * math.log(5) + math.log(20)) / 200)
    assert result.primary.value.item() == pytest.approx(expected)
    assert certificate.max_logit_error.item() == 0
    assert certificate.parameter_count == 4 and certificate.values_per_parameter == 5
    assert "matrix_project" in certificate.certificate_kind
    with torch.no_grad():
        projected.model.weight.add_(0.125)
    with pytest.raises(InputError, match="finite code"):
        prepare_arora_projection(
            model,
            projected,
            data,
            margin=0.5,
            helper_is_data_independent=True,
            alphabet_is_data_independent=True,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "forward_buffer",
        "forward",
        "hook",
        "pre_hook",
        "buffer",
        "nonpersistent_buffer",
        "add_layer",
        "replace_relu",
        "tied",
        "dtype",
    ],
)
def test_arora_certificate_revalidates_entire_predictor(mutation):
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(2, 2, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        model[2].weight.copy_(torch.eye(2, dtype=torch.float64))
    helper = torch.tensor(
        [
            [[1, 1], [1, 1]],
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]],
        ],
        dtype=torch.float64,
    )
    projected = AroraMatrixProject(
        {"0.weight": helper, "2.weight": helper},
        torch.arange(-2, 3, dtype=torch.float64),
    ).project(model)
    predictor = projected.model
    if mutation == "forward_buffer":
        # Previously accepted q/r while the real predictor lived in a buffer.
        predictor.register_buffer(
            "learned_hidden_predictor", torch.eye(2, dtype=torch.float64)
        )
        predictor.forward = lambda x: x @ predictor.learned_hidden_predictor.T
    elif mutation == "forward":
        predictor.forward = lambda x: x
    elif mutation == "hook":
        predictor[0].register_forward_hook(lambda module, args, output: output + 1)
    elif mutation == "pre_hook":
        predictor[0].register_forward_pre_hook(lambda module, args: (args[0] + 1,))
    elif mutation in ("buffer", "nonpersistent_buffer"):
        predictor[0].register_buffer(
            "unaccounted_state", torch.eye(2), persistent=mutation == "buffer"
        )
    elif mutation == "add_layer":
        predictor.add_module("unaccounted_layer", nn.Identity())
    elif mutation == "replace_relu":
        predictor[1] = nn.Identity()  # Still a whitelisted type, but not bound schema.
    elif mutation == "tied":
        predictor[2].weight = predictor[0].weight
    elif mutation == "dtype":
        predictor.to(torch.float32)
    data = DataSource.from_tensors(
        torch.eye(2, dtype=torch.float64).repeat(3, 1), torch.tensor([0, 1]).repeat(3)
    )
    with pytest.raises(InputError, match="invalid Arora finite predictor"):
        prepare_arora_projection(
            model,
            projected,
            data,
            margin=0.5,
            helper_is_data_independent=True,
            alphabet_is_data_independent=True,
        )
