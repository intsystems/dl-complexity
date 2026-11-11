"""prepare_fit_statistics with frozen, tied and train-mode layers (C-12 - C-14)."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity import DataSource, ModelContext
from dl_complexity.criteria import AIC, BIC, HQIC
from dl_complexity.errors import InputError
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    GaussianLikelihood,
    prepare_fit_statistics,
)

DT = torch.float64


def classification_data(n=40, d=3, classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=DT)
    y = torch.randint(classes, (n,), generator=g)
    return x, y


def partly_frozen_model():
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(3, 4, dtype=DT),  # 12 + 4
        nn.BatchNorm1d(4, dtype=DT),  # 4 + 4 affine; running stats are buffers
        nn.ReLU(),
        nn.Linear(4, 2, dtype=DT),  # 8 + 2
    )
    model[0].bias.requires_grad_(False)
    model[3].weight.requires_grad_(False)
    return model


# trainable vs all


def test_frozen_parameters_are_excluded_only_from_trainable_count():
    model = partly_frozen_model()
    x, y = classification_data()
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    all_k = 12 + 4 + 4 + 4 + 8 + 2
    trainable_k = all_k - 4 - 8
    assert prepare_fit_statistics(ctx, data=data).k == trainable_k  # default
    assert (
        prepare_fit_statistics(ctx, data=data, parameter_count="trainable").k
        == trainable_k
    )
    assert prepare_fit_statistics(ctx, data=data, parameter_count="all").k == all_k
    assert prepare_fit_statistics(ctx, data=data, parameter_count=5).k == 5
    # BatchNorm buffers are not parameters
    assert sum(b.numel() for b in model.buffers()) == 4 + 4 + 1


@pytest.mark.parametrize("policy,k", [("trainable", 22), ("all", 34), (7, 7)], ids=str)
def test_context_route_penalty_uses_selected_parameter_count(policy, k):
    model = partly_frozen_model()
    x, y = classification_data(n=40)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    with torch.no_grad():
        n = len(y)
        bic = BIC(parameter_count=policy).estimate(ctx, data=data)
        aic = AIC(parameter_count=policy).estimate(ctx, data=data)
        hqic = HQIC(parameter_count=policy).estimate(ctx, data=data)
    torch.testing.assert_close(
        bic.quantities["penalty"].value, torch.tensor(k * math.log(n), dtype=DT)
    )
    torch.testing.assert_close(
        aic.quantities["penalty"].value, torch.tensor(2.0 * k, dtype=DT)
    )
    torch.testing.assert_close(
        hqic.quantities["penalty"].value,
        torch.tensor(2 * k * math.log(math.log(n)), dtype=DT),
    )
    assert bic.config["k"] == aic.config["k"] == hqic.config["k"] == k


def test_fully_frozen_model_has_zero_trainable_parameters():
    model = nn.Linear(2, 1, dtype=DT).requires_grad_(False)
    x = torch.randn(5, 2, dtype=DT)
    stats = prepare_fit_statistics(
        ModelContext(model, likelihood=GaussianLikelihood()),
        data=DataSource.from_tensors(x, torch.zeros(5, 1, dtype=DT)),
    )
    assert stats.k == 0


# tied weights


class TiedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(3, 3, bias=False, dtype=DT)  # 9, shared
        self.dec = nn.Linear(3, 2, bias=True, dtype=DT)  # weight 6 + bias 2
        self.head = nn.Linear(3, 3, bias=False, dtype=DT)
        self.head.weight = self.enc.weight  # tied: same Parameter object
        self.alias = self.enc.weight  # registered a third time under another name

    def forward(self, x):
        return self.dec(torch.relu(self.head(self.enc(x))))


def test_tied_parameters_are_counted_once():
    model = TiedAutoencoder()
    names = [name for name, _ in model.named_parameters(remove_duplicate=False)]
    assert (
        names.count("enc.weight") == 1 and "head.weight" in names and "alias" in names
    )
    x, y = classification_data(n=10)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    unique_k = 9 + 6 + 2
    assert prepare_fit_statistics(ctx, data=data).k == unique_k
    assert prepare_fit_statistics(ctx, data=data, parameter_count="all").k == unique_k
    model.enc.weight.requires_grad_(False)  # freezes every alias at once
    assert prepare_fit_statistics(ctx, data=data).k == 6 + 2


# eval mode


def stochastic_model():
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(3, 8, dtype=DT),
        nn.BatchNorm1d(8, dtype=DT),
        nn.ReLU(),
        nn.Dropout(p=0.9),
        nn.Linear(8, 4, dtype=DT),
    )
    with torch.no_grad():  # running stats far from the batch statistics
        model[1].running_mean.copy_(torch.linspace(-2, 2, 8, dtype=DT))
        model[1].running_var.copy_(torch.linspace(0.5, 3, 8, dtype=DT))
    return model.train()


def eval_mode_log_likelihood(model, x, y):
    reference = copy.deepcopy(model).eval()
    with torch.no_grad():
        return -F.cross_entropy(reference(x), y, reduction="sum")


def test_dropout_and_batchnorm_are_evaluated_in_eval_mode():
    model = stochastic_model()
    x, y = classification_data(n=64, d=3, classes=4, seed=5)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    expected = eval_mode_log_likelihood(model, x, y)
    with torch.no_grad():
        first = prepare_fit_statistics(ctx, data=data).log_likelihood
        second = prepare_fit_statistics(ctx, data=data).log_likelihood
    torch.testing.assert_close(first, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(first, second)  # no dropout noise between calls
    # make sure train mode would really give something else
    with torch.no_grad():
        train_ll = -F.cross_entropy(copy.deepcopy(model).train()(x), y, reduction="sum")
    assert not torch.allclose(train_ll, expected)


def test_eval_mode_result_does_not_depend_on_batching_with_batchnorm():
    # in train mode BatchNorm would depend on the batch split
    model = stochastic_model()
    x, y = classification_data(n=30, d=3, classes=4, seed=6)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    batches = [(x[:7], y[:7]), (x[7:19], y[7:19]), (x[19:], y[19:])]
    with torch.no_grad():
        whole = prepare_fit_statistics(ctx, data=DataSource.from_tensors(x, y))
        split = prepare_fit_statistics(ctx, data=DataSource(batches, n_observations=30))
    torch.testing.assert_close(split.log_likelihood, whole.log_likelihood)
    assert split.n == whole.n == 30


def test_bic_context_route_matches_manual_eval_mode_formula():
    model = stochastic_model()
    x, y = classification_data(n=50, d=3, classes=4, seed=7)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    with torch.no_grad():
        result = BIC().estimate(ctx, data=DataSource.from_tensors(x, y))
    k = sum(p.numel() for p in model.parameters())
    expected = -2 * eval_mode_log_likelihood(model, x, y) + k * math.log(50)
    torch.testing.assert_close(result.primary.value, expected, rtol=1e-12, atol=1e-10)
    assert model.training and model[3].training  # caller's mode is untouched


# errors


def linear_context(likelihood=None):
    return ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=likelihood or CategoricalLikelihood()
    )


@pytest.mark.parametrize(
    "prediction,target",
    [
        (torch.zeros(4, dtype=DT), torch.zeros(4, dtype=torch.long)),  # no class axis
        (torch.zeros(4, 3, dtype=DT), torch.zeros(5, dtype=torch.long)),  # length
        (torch.zeros(4, 3, dtype=DT), torch.zeros(4, dtype=DT)),  # float labels
    ],
    ids=["logits_1d", "length_mismatch", "float_targets"],
)
def test_categorical_likelihood_rejects_malformed_arguments(prediction, target):
    with pytest.raises(InputError):
        CategoricalLikelihood().log_prob(prediction, target)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_likelihood_rejects_invalid_noise_scale(sigma):
    with pytest.raises(InputError):
        GaussianLikelihood(sigma=sigma)


def test_gaussian_likelihood_does_not_broadcast_n_by_one_against_n():
    # [N,1] - [N] would broadcast to [N,N]
    with pytest.raises(InputError):
        GaussianLikelihood().log_prob(
            torch.zeros(5, 1, dtype=DT), torch.zeros(5, dtype=DT)
        )


def test_prepare_fit_statistics_requires_likelihood_and_data_source():
    x, y = classification_data(n=6, d=2, classes=3)
    data = DataSource.from_tensors(x, y)
    with pytest.raises(InputError, match="likelihood"):
        prepare_fit_statistics(ModelContext(nn.Linear(2, 3, dtype=DT)), data=data)
    with pytest.raises(InputError, match="likelihood"):
        prepare_fit_statistics(None, data=data)
    with pytest.raises(InputError, match="DataSource"):
        prepare_fit_statistics(linear_context(), data=[(x, y)])
    with pytest.raises(InputError, match="parameter_count"):
        prepare_fit_statistics(linear_context(), data=data, parameter_count="unique")


@pytest.mark.parametrize(
    "batches",
    [
        [torch.zeros(3, 2, dtype=DT)],  # not an (inputs, targets) pair
        [(torch.zeros(3, 2, dtype=DT), torch.zeros(3, dtype=torch.long), None)],
    ],
    ids=["bare_tensor", "triple"],
)
def test_batches_must_be_input_target_pairs(batches):
    with pytest.raises(InputError, match="batch"):
        prepare_fit_statistics(linear_context(), data=DataSource(batches))


def test_reduced_loss_is_not_accepted_as_a_pointwise_likelihood():
    def mean_log_prob(prediction, target):  # what a typical loss returns
        return CategoricalLikelihood().log_prob(prediction, target).mean()

    x, y = classification_data(n=6, d=2, classes=3)
    ctx = ModelContext(nn.Linear(2, 3, dtype=DT), likelihood=mean_log_prob)
    with pytest.raises(InputError, match="pointwise"):
        prepare_fit_statistics(ctx, data=DataSource.from_tensors(x, y))


def test_nonfinite_pointwise_likelihood_is_rejected():
    model = nn.Linear(2, 1, dtype=DT)
    with torch.no_grad():
        model.weight.fill_(float("inf"))
    x = torch.ones(4, 2, dtype=DT)
    ctx = ModelContext(model, likelihood=GaussianLikelihood())
    with pytest.raises(InputError, match="nonfinite"):
        prepare_fit_statistics(
            ctx, data=DataSource.from_tensors(x, torch.zeros(4, 1, dtype=DT))
        )


def test_model_that_cannot_be_isolated_is_rejected_without_mutation():
    import threading

    class WithLock(nn.Linear):
        def __init__(self):
            super().__init__(2, 1, dtype=DT)
            self.lock = threading.Lock()  # not deep-copyable

    model = WithLock()
    before = model.weight.detach().clone()
    x = torch.ones(3, 2, dtype=DT)
    ctx = ModelContext(model, likelihood=GaussianLikelihood())
    with pytest.raises(InputError, match="isolated"):
        prepare_fit_statistics(
            ctx, data=DataSource.from_tensors(x, torch.zeros(3, 1, dtype=DT))
        )
    assert torch.equal(model.weight, before)
