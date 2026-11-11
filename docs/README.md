# DL Complexity: документация

_Оценки сложности PyTorch-моделей: информационные критерии, байесовские оценки,
кодовые длины, границы риска и статистики весов._

---

## 🚀 Начало работы

Требуются Python 3.11+ и PyTorch 2.2+. Для установки из каталога исходников:

```bash
python -m pip install .
```

Пакет импортируется как `dl_complexity`. Ниже описаны поддерживаемые варианты
методов и требования к их входам.

Пример оценивает линейную регрессию с известным стандартным отклонением шума.
Для данных `y = 2x` коэффициент `2` максимизирует гауссовское правдоподобие
при фиксированном `sigma=1`. Оценивается один параметр, свободного члена нет.

```python
import torch
from torch import nn

from dl_complexity import ComplexitySuite, DataSource, ModelContext
from dl_complexity.criteria import AIC, BIC, HQIC
from dl_complexity.integrations.torch import (
    GaussianLikelihood,
    prepare_fit_statistics,
)

model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
with torch.no_grad():
    model.weight.fill_(2.0)

x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
data = DataSource.from_tensors(x, 2.0 * x)
context = ModelContext(model, likelihood=GaussianLikelihood(sigma=1.0))

# Один проход по данным для трёх критериев.
with torch.no_grad():
    statistics = prepare_fit_statistics(context, data=data, parameter_count=1)

suite = ComplexitySuite({"aic": AIC(), "bic": BIC(), "hqic": HQIC()})
outcomes = suite.estimate(inputs={name: statistics for name in ("aic", "bic", "hqic")})
for name, outcome in outcomes.items():
    value = outcome.result.primary
    print(name, round(value.value.item(), 6), value.unit)

# aic 7.513631 native_ic
# bic 6.612243 native_ic
# hqic 5.701727 native_ic
```

Значение `native_ic` сохраняет исходную шкалу критерия. Оно не переводится в
bits или в вероятность ошибки. Компоненты оценки находятся в
`result.quantities`, условия применения — в `result.assumptions`.

## 🧭 Выбор метода

| Что нужно оценить | Методы | Что передать |
| --- | --- | --- |
| Штраф за размерность статистической модели | AIC, BIC, HQIC | Модель с likelihood и данными либо `FitStatistics` |
| Posterior predictive качество с поправкой на сложность | WAIC | Матрицу log-likelihood `[draws, observations]` обычного posterior |
| Приближение отрицательного log evidence | WBIC | Total log-likelihood по tempered posterior при `beta=1/ln(n)` |
| Вариационную свободную энергию | `VariationalCode` | Ожидаемый total NLL и KL либо явную подготовку из `q` и prior |
| Laplace-приближение отрицательного log evidence | `KFACLaplace` | Likelihood, нормированный prior и logdet либо KFAC-факторы |
| Двухчастную длину описания | `TwoPartMDL` | Длину модели и NLL декодированного предиктора |
| Стоимость последовательного предсказания | Blockwise, Online, Replay prequential | Журнал кодирования с соблюдением порядка наблюдений |
| Риск сжатого предиктора | `AroraCompressionBound` | Конечное семейство и сертификат ошибки сжатия на всей выборке |
| Нормы, разреженность и размер весов | `WeightStatistics` | Модель или выбранные тензоры весов |

По обычному checkpoint можно восстановить `ModelContext`, если известна
архитектура. Для норм достаточно весов. Для AIC/BIC/HQIC дополнительно нужны
likelihood, данные и обоснованное число параметров. Один checkpoint не содержит
posterior draws, префиксную историю обучения или сертификат Arora.

Перед вычислением можно вызвать `method.check(...)`. Статус `READY` означает
достаточность входов, но не подтверждает статистические предпосылки метода.
Вызов `estimate` не обучает модель и не запускает сэмплер.

## 📚 Руководства

- [Использование и вычисления](CODE_WALKTHROUGH.md): контекст, checkpoint,
  общий вызов, формулы, память, autograd и ограничения методов.
- [Справочник публичного API](API_REFERENCE.md): импорты, сигнатуры,
  параметры, возвращаемые типы и ошибки.
- [Структура исходников](../src/dl_complexity/README.md): базовые интерфейсы,
  связи модулей и добавление оценщика.
- [Численные проверки](../tests/README.md): аналитические примеры,
  ожидаемые значения, допуски и запуск тестов.
- [Полный пример](../examples/quickstart.py): совместная оценка PyTorch-модели
  и работа с недостающими входами.
- [Литература](../literature/README.md): первоисточники методов.

## Требования к входам и варианты методов

- WAIC использует вариант WAIC2 с выборочной дисперсией `ddof=1`.
  WBIC принимает готовые tempered draws, собственного сэмплера нет.
- Подготовка точных KFAC-факторов поддерживает один встроенный `nn.Linear`
  с гауссовским likelihood. Для произвольной сети факторы нужно предоставить.
- Arora включает Matrix-Project и условную Theorem 2.1. Полная noise-stability
  Theorem 4.1 и её геометрические условия не реализованы.
- `ModelContext.structure` описывает модули и параметры, не forward-граф.
  Loss и likelihood задаются отдельно, их соответствие указывает пользователь.
- `WeightStatistics.parameter_count` — тензор в dtype весов, который может
  округлять целые числа. Для точного счёта используйте
  `sum(p.numel() for p in model.parameters())`. `FitStatistics.k` хранится как Python int.
- Бинарный кодек учитывает заголовок и все байты сообщения, но упаковка битов
  использует растущие Python integers. Её стоимость может быть квадратичной
  по длине сообщения. Не загружайте недоверенные payload без внешних ограничений
  ресурсов.
