"""Population KL under finite isotropic design and unit Gaussian response noise."""

import numpy as np
from common import save

SUPPORT = np.sqrt(2) * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
TEACHER_U = np.array([[1.0], [0.0]])
TEACHER_V = np.array([[1.0, 0.0]])
MLP_U = np.array([[0.8, -0.4, 0.6], [-0.3, 0.9, 0.5]])
MLP_V = np.array([[1.0], [-0.7], [0.5]])


def experiment(out, steps, repeats, seed):
    metrics, fits = [], []
    checkpoints = sorted(set(np.geomspace(1, steps, 50).round().astype(int)))
    for model in ("linear", "rank1", "rank0", "mlp"):
        scalar = model == "linear"
        mlp = model == "mlp"
        support = np.array([[-1.0], [1.0]]) if scalar else SUPPORT
        if scalar:
            truth = support * 0.8
        elif mlp:
            truth = np.tanh(support @ MLP_U) @ MLP_V
        else:
            truth = support @ TEACHER_U @ TEACHER_V * (model == "rank1")
        for batch in (1, 8):
            for policy in ("constant", "inverse"):
                # Same exogenous data across step schedules and paired starts.
                rng = np.random.default_rng(seed)
                w = np.zeros((repeats, 1, 1))
                u0 = MLP_U if mlp else TEACHER_U
                v0 = MLP_V if mlp else TEACHER_V
                scale = 0.3 if model == "rank0" else 1.0
                u = scale * u0 + rng.normal(0, 0.15, (repeats, *u0.shape))
                v = scale * v0 + rng.normal(0, 0.15, (repeats, *v0.shape))
                total = np.zeros(repeats)
                curves = []
                for t in range(1, steps + 1):
                    if scalar:
                        pred = support @ w
                    elif mlp:
                        pred = np.tanh(support @ u) @ v
                    else:
                        pred = support @ (u @ v)
                    residual = pred - truth
                    kl = 0.5 * np.sum(residual**2, axis=2).mean(axis=1)
                    total += kl
                    if t in checkpoints:
                        curves.append(total.copy())
                        for rep, (risk, cum) in enumerate(zip(kl, total)):
                            metrics.append(
                                dict(
                                    model=model,
                                    batch=batch,
                                    policy=policy,
                                    replicate=rep,
                                    T=t,
                                    pre_update_kl_nats=risk,
                                    cum_kl_nats=cum,
                                    batch_cum_kl_nats=batch * cum,
                                )
                            )
                    ix = rng.integers(len(support), size=(repeats, batch))
                    x = support[ix]
                    err = residual[np.arange(repeats)[:, None], ix] - rng.normal(
                        size=(*ix.shape, truth.shape[1])
                    )
                    eta = 0.035 if policy == "constant" else 1 / (t + 9)
                    if scalar:
                        w -= eta * (x.swapaxes(1, 2) @ err) / batch
                    else:
                        h = x @ u
                        if mlp:
                            h = np.tanh(h)
                        back = err @ v.swapaxes(1, 2)
                        if mlp:
                            back *= 1 - h**2
                        u -= eta * (x.swapaxes(1, 2) @ back) / batch
                        v -= eta * (h.swapaxes(1, 2) @ err) / batch
                time = np.array(checkpoints)
                values = np.array(curves)
                for scale_name, axis in [("log", np.log(time)), ("linear", time)]:
                    train = (time >= steps / 8) & (time <= steps / 2)
                    test = time > steps / 2
                    slope, intercept = np.polyfit(axis[train], values[train], 1)
                    error = values[test] - (axis[test, None] * slope + intercept)
                    fits.append(
                        dict(
                            model=model,
                            batch=batch,
                            policy=policy,
                            fit=scale_name,
                            fitted_slope=slope.mean(),
                            slope_se=slope.std(ddof=1) / np.sqrt(repeats),
                            heldout_rmse_mean_curve=np.sqrt(
                                np.mean(error.mean(axis=1) ** 2)
                            ),
                            lambda_reference={
                                "linear": 0.5,
                                "rank1": 1.5,
                                "rank0": 1.0,
                                "mlp": "",
                            }[model],
                        )
                    )
    save(out, "sgd", metrics)
    save(out, "sgd_fits", fits)
