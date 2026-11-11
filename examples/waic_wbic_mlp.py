"""WAIC and WBIC for MLPs of different width, compared with AIC and BIC.

Labels come from a teacher net 2-8-3 (tanh). Students are trained to the MAP
with prior N(0, I), then MALA chains start from those weights.
AIC/2n and WAIC/2n estimate the test loss, BIC/2n and WBIC/n the free energy.
lambda_hat = (WBIC - nL_n(w0)) / log n with w0 = teacher, so only for
width >= 8 (narrower nets cannot represent the teacher). AIC and BIC are
taken at the MAP, not at the MLE.

Run: PYTHONPATH=src python examples/waic_wbic_mlp.py [--json out.json]
"""

import argparse
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity import DataSource, ModelContext
from dl_complexity.bayesian import WBIC
from dl_complexity.criteria import AIC, BIC, WAIC, PosteriorLogLikelihood
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    Langevin,
    prepare_fit_statistics,
    prepare_tempered_likelihood,
    sample_posterior,
)

DT = torch.float64
N_TRAIN, N_TEST, ALPHA = 600, 20_000, 1.0
TEACHER_WIDTH = 8
WIDTHS = (1, 2, 4, 8, 16, 32)


def teacher_logits(x):
    # fixed teacher, weights from its own seed
    weights = torch.Generator().manual_seed(1000)
    w1 = 2.0 * torch.randn(2, TEACHER_WIDTH, generator=weights, dtype=DT)
    b1 = torch.randn(TEACHER_WIDTH, generator=weights, dtype=DT)
    w2 = 1.5 * torch.randn(TEACHER_WIDTH, 3, generator=weights, dtype=DT)
    return torch.tanh(x @ w1 + b1) @ w2


def teacher_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=DT)
    probabilities = F.softmax(teacher_logits(x), -1)
    y = torch.multinomial(probabilities, 1, generator=g).squeeze(-1)
    return x, y


def train_map(width, x, y, steps=3000, seed=0):
    # MAP: NLL + alpha/2 |w|^2, full-batch Adam
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, width, dtype=DT), nn.Tanh(), nn.Linear(width, 3, dtype=DT)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        penalty = sum(p.square().sum() for p in model.parameters())
        loss = F.cross_entropy(model(x), y, reduction="sum") + 0.5 * ALPHA * penalty
        (loss / len(y)).backward()
        optimizer.step()
    return model


def tune_step(context, data, beta, seed):
    # largest step with acceptance >= 0.5 on a short pilot run
    for step in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        pilot = sample_posterior(
            context,
            data=data,
            sampler=Langevin(step, 100, burn_in=100, metropolis=True, seed=seed),
            prior_precision=ALPHA,
            beta=beta,
        )
        if pilot.acceptance_rate >= 0.5:
            return step, pilot.acceptance_rate
    return step, pilot.acceptance_rate


def bayes_test_loss(model, draws, x, y):
    # test loss of the posterior predictive, running logsumexp over draws
    shapes = {name: p.shape for name, p in model.named_parameters()}
    total = None
    for vector in draws.parameters:
        state, offset = {}, 0
        for name in draws.names:
            size = math.prod(shapes[name])
            state[name] = vector[offset : offset + size].reshape(shapes[name])
            offset += size
        logits = torch.func.functional_call(model, state, (x,))
        log_p = F.log_softmax(logits, -1).gather(-1, y[:, None]).squeeze(-1)
        total = log_p if total is None else torch.logaddexp(total, log_p)
    return float(-(total - math.log(len(draws.parameters))).mean())


def evaluate(width, x, y, x_test, y_test, teacher_train_nll):
    model = train_map(width, x, y)
    context = ModelContext(
        model, likelihood=CategoricalLikelihood(), state_id=f"mlp-{width}-map"
    )
    data = DataSource.from_tensors(x, y, data_id="teacher-train-600")
    n = len(y)
    with torch.no_grad():
        statistics = prepare_fit_statistics(context, data=data)
        map_test = float(F.cross_entropy(model(x_test), y_test))
        aic = float(AIC().estimate(inputs=statistics).primary.value)
        bic = float(BIC().estimate(inputs=statistics).primary.value)
    k = statistics.k

    step, _ = tune_step(context, data, 1.0, seed=1)
    posterior = sample_posterior(
        context,
        data=data,
        sampler=Langevin(step, 1000, burn_in=1000, thin=2, metropolis=True, seed=2),
        prior_precision=ALPHA,
    )
    waic = WAIC().estimate(inputs=PosteriorLogLikelihood(posterior.log_likelihood))
    with torch.no_grad():
        bayes_test = bayes_test_loss(model, posterior, x_test, y_test)

    beta = 1 / math.log(n)
    tempered_step, tempered_pilot_rate = tune_step(context, data, beta, seed=3)
    tempered = prepare_tempered_likelihood(
        context,
        data=data,
        sampler=Langevin(
            tempered_step, 1000, burn_in=1000, thin=2, metropolis=True, seed=4
        ),
        prior_precision=ALPHA,
    )
    wbic = float(WBIC().estimate(inputs=tempered).primary.value)
    realizable = width >= TEACHER_WIDTH
    return {
        "width": width,
        "k": k,
        "train_nll_map_per_obs": -float(statistics.log_likelihood) / n,
        "test_loss_map": map_test,
        "test_loss_bayes": bayes_test,
        "aic_per_2n": aic / (2 * n),
        "waic_per_2n": float(waic.primary.value) / (2 * n),
        "p_waic": float(waic.quantities["p_waic"].value),
        "acceptance_waic_chain": posterior.acceptance_rate,
        "bic_per_2n": bic / (2 * n),
        "wbic_per_n": wbic / n,
        "pilot_acceptance_wbic": tempered_pilot_rate,
        "lambda_hat": (
            (wbic - teacher_train_nll) / math.log(n) if realizable else None
        ),
        "half_dimension": k / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="save the results table to this file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, y = teacher_data(N_TRAIN, seed=1)
    x_test, y_test = teacher_data(N_TEST, seed=2)
    teacher_train_nll = float(F.cross_entropy(teacher_logits(x), y, reduction="sum"))
    teacher_test = float(F.cross_entropy(teacher_logits(x_test), y_test))
    rows = []
    started = time.time()
    header = (
        f"{'width':>5} {'k':>4} | {'test MAP':>8} {'test Bayes':>10} "
        f"{'AIC/2n':>7} {'WAIC/2n':>8} {'p_waic':>6} {'acc':>4} | "
        f"{'BIC/2n':>7} {'WBIC/n':>7} | {'lambda_hat':>10} {'d/2':>6}"
    )
    print(
        f"n = {N_TRAIN}, prior N(0, I), teacher 2-{TEACHER_WIDTH}-3 tanh "
        f"(its test loss {teacher_test:.4f}). Nats per observation, except "
        "p_waic, acc, lambda_hat and d/2."
    )
    print(header)
    print("-" * len(header))
    for width in WIDTHS:
        row = evaluate(width, x, y, x_test, y_test, teacher_train_nll)
        rows.append(row)
        lam = "-" if row["lambda_hat"] is None else f"{row['lambda_hat']:.1f}"
        print(
            f"{row['width']:>5} {row['k']:>4} | {row['test_loss_map']:>8.4f} "
            f"{row['test_loss_bayes']:>10.4f} {row['aic_per_2n']:>7.4f} "
            f"{row['waic_per_2n']:>8.4f} {row['p_waic']:>6.1f} "
            f"{row['acceptance_waic_chain']:>4.2f} | {row['bic_per_2n']:>7.4f} "
            f"{row['wbic_per_n']:>7.4f} | {lam:>10} {row['half_dimension']:>6.1f}",
            flush=True,
        )
    print(f"done in {time.time() - started:.0f} s")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
