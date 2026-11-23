"""Finite-horizon exact schedule search and large-n geometric experiments."""

from functools import lru_cache
import numpy as np
from scipy.stats import binom, chi2
from scipy.special import betaln, xlogy
from scipy.optimize import minimize_scalar
from common import LN2, KAPPA, bern_kl, bound, pfr, delta, read_delta, replay, save

# w uniform on [0,1]. First p=w, then p=w**4.
MODELS = [("regular", 0.37, 1.0, 1), ("quartic", 0.0, 0.25, 4)]


def experiment(out, limit, repeats, seed):
    rows, schedules, asymptotic, messages = [], [], [], []
    for model, q, a, power in MODELS:

        @lru_cache(None)
        def law(n):
            if q == 0 or n == 0:
                k = np.array([0])
            else:
                lo, hi = binom.ppf([1e-14, 1 - 1e-14], n, q).astype(int)
                k = np.arange(lo, hi + 1)
            weight = binom.pmf(k, n, q)
            weight /= weight.sum()
            pred = (k + a) / (n + a + 1)
            return k, weight, bern_kl(q, pred) / LN2

        def edge(n, batches):
            _, weight, risk = law(n)
            d = np.asarray(batches)[..., None] * risk
            return np.sum(bound(d) * weight, axis=-1)

        best = np.full(limit + 1, np.inf)
        prev = np.zeros(limit + 1, dtype=int)
        best[0] = 0
        unit_cost = redundancy = plug_redundancy = 0.0
        for n in range(limit + 1):
            k, weight, d = law(n)
            risk = np.dot(weight, d)
            neg_log_mix = -(betaln(k + a, n - k + 1) - betaln(a, 1)) / LN2
            entropy = -(xlogy(q, q) + xlogy(1 - q, 1 - q)) / LN2
            bayes = np.dot(weight, neg_log_mix)
            np.testing.assert_allclose(bayes - n * entropy, redundancy, atol=1e-8)
            plug = (k + 0.5) / (n + 1)
            rows.append(
                dict(
                    model=model,
                    n=n,
                    final_kl_nats=risk * LN2,
                    bayes_data_bits=bayes,
                    teacher_entropy_bits=n * entropy,
                    preq_beta_half_data_bits=n * entropy + plug_redundancy,
                    preq_beta_half_next_kl_bits=weight @ (bern_kl(q, plug) / LN2),
                    bayes_redundancy_bits=redundancy,
                    unit_rhs_bits=unit_cost,
                    optimal_rhs_bits=best[n],
                    previous_n=prev[n],
                )
            )
            if n == limit:
                break
            batches = np.arange(1, limit - n + 1)
            costs = edge(n, batches)
            ids = n + batches
            better = best[n] + costs < best[ids]
            best[ids[better]] = best[n] + costs[better]
            prev[ids[better]] = n
            redundancy += risk
            plug_redundancy += weight @ (bern_kl(q, plug) / LN2)
            unit_cost += costs[0]

        def quant(eps):
            for bits in range(32):
                levels = 2**bits
                w = q ** (1 / power)
                loc = min(levels - 1, int(w * levels))
                p = ((loc + 0.5) / levels) ** power
                if bern_kl(q, p) <= eps:
                    return bits, p
            raise ValueError("grid insufficient")

        model_rows = rows[-limit - 1 :]
        for eps in [2.0**-j for j in range(3, 17)]:
            qb, qp = quant(eps)

            def record(policy, sizes, cost, risk):
                schedules.append(
                    dict(
                        model=model,
                        epsilon_nats=eps,
                        policy=policy,
                        T=len(sizes),
                        n=sum(sizes),
                        batch_sizes=" ".join(map(str, sizes)),
                        rhs_bits=cost,
                        final_kl_nats=risk,
                        quant_payload_bits=qb,
                        quant_total_bits=qb + len(delta(qb + 1)),
                        quant_kl_nats=bern_kl(q, qp),
                    )
                )

            feasible = [r for r in model_rows if r["final_kl_nats"] <= eps]
            if feasible:
                r = min(feasible, key=lambda r: r["optimal_rhs_bits"])
                n = r["n"]
                sizes = []
                while n:
                    sizes.append(n - prev[n])
                    n = prev[n]
                record("DP", sizes[::-1], r["optimal_rhs_bits"], r["final_kl_nats"])
            for growth in (1.5, 2.0, 4.0, 8.0):
                sizes, n, cost = [], 0, 0.0
                while True:
                    _, w, d = law(n)
                    risk = w @ d * LN2
                    if risk <= eps:
                        break
                    b = max(1, round((growth - 1) * n))
                    sizes.append(b)
                    cost += edge(n, b)
                    n += b
                record(f"geom{growth}", sizes, cost, risk)

        # n*KL in nats converges to chi-square(1)/2, or deterministically to 1/4.
        def coefficient(log_ratio):
            x = np.expm1(log_ratio)
            cost = (
                chi2.expect(lambda z: bound(x * z / (2 * LN2)), args=(1,))
                if q
                else bound(x * a / LN2)
            )
            return cost / log_ratio

        opt = minimize_scalar(coefficient, bounds=(0.1, 5), method="bounded")
        asymptotic.append(
            dict(
                model=model,
                lambda_value=0.5 if q else a,
                geometric_ratio=np.exp(opt.x),
                bits_per_ln_inverse_epsilon=opt.fun,
            )
        )
        for r in range(repeats):
            for name, sizes in [("unit", [1] * 16), ("growing", [1, 2, 4, 8])]:
                stream, k, n, cost = "", 0, 0, 0.0
                for t, b in enumerate(sizes):
                    p = (k + a) / (n + a + 1)
                    j, sample = pfr(seed + r, t, p, q, b)
                    stream += delta(j)
                    cost += bound(b * bern_kl(q, p) / LN2)
                    k += sample.sum()
                    n += b
                pos = kd = nd = 0
                for t, b in enumerate(sizes):
                    j, pos = read_delta(stream, pos)
                    kd += replay(seed + r, t, (kd + a) / (nd + a + 1), b, j).sum()
                    nd += b
                assert (kd, nd, pos) == (k, n, len(stream))
                messages.append(
                    dict(
                        model=model,
                        seed=seed + r,
                        policy=name,
                        T=len(sizes),
                        n=n,
                        ones=k,
                        message=stream,
                        actual_bits=pos,
                        rhs_bits=cost,
                    )
                )
    save(out, "bayes", rows)
    save(out, "schedules", schedules)
    save(out, "geometric_limit", asymptotic)
    save(out, "pfr", messages)
