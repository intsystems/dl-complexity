"""Small API example, not a fitted-model experiment or a benchmark.

Run after installation, or from the checkout with PYTHONPATH=src.
The zero-weight model below is not claimed to be an MLE.
"""

import torch
from torch import nn

from dl_complexity import ComplexitySuite, DataSource, ModelContext
from dl_complexity.coding import TwoPartMDL, prepare_two_part
from dl_complexity.compression import BinaryModelCodec
from dl_complexity.criteria import AIC, BIC, HQIC, WAIC
from dl_complexity.diagnostics import WeightStatistics
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    prepare_fit_statistics,
)


def main():
    model = nn.Linear(1, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    context = ModelContext(
        model, likelihood=CategoricalLikelihood(), state_id="example-zero-weights"
    )
    data = DataSource.from_tensors(
        torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64),
        torch.tensor([0, 0, 1, 1]),
        data_id="example-four-observations",
    )
    # One likelihood pass can be explicitly shared between classical criteria.
    statistics = prepare_fit_statistics(context, data=data)
    coded = prepare_two_part(context, data=data, codec=BinaryModelCodec())
    suite = ComplexitySuite(
        {
            "aic": AIC(),
            "bic": BIC(),
            "hqic": HQIC(),
            "waic": WAIC(),
            "two_part": TwoPartMDL(),
            "weights": WeightStatistics(),
        },
        errors="collect",
    )
    report = suite.estimate(
        data=data,
        inputs={
            "aic": statistics,
            "bic": statistics,
            "hqic": statistics,
            "two_part": coded,
        },
        context=context,
    )
    for name, outcome in report.items():
        if outcome.error is not None:
            raise outcome.error
        if outcome.result is None:
            print(name, outcome.requirements.status.value, outcome.requirements.missing)
        elif outcome.result.primary is None:
            print(name, outcome.result.kind, "named diagnostics")
        else:
            quantity = outcome.result.primary
            print(
                name, float(quantity.value.detach()), quantity.unit, outcome.result.kind
            )


if __name__ == "__main__":
    main()
