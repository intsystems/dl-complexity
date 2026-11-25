# Как работает DL Complexity: функции и параметры

Краткая шпаргалка для знакомства с кодом и выступления. Полные сигнатуры —
в [справочнике API](api/index.rst), рабочие результаты — в
[демонстрационном ноутбуке](../examples/demo.ipynb).

## 1. Что делает библиотека

Библиотека принимает модель PyTorch и данные либо заранее рассчитанные численные
входы. Затем вычисляет выбранную оценку сложности и возвращает её вместе
с единицами, составляющими и допущениями.

```text
Модель + likelihood + данные         Готовые численные входы
             |                               |
             +---------- check --------------+
                           |
                        estimate
                           |
                   ComplexityResult
```

`likelihood` задаёт вероятностную модель наблюдений. Это необходимо для вычисления
log likelihood и NLL — отрицательного логарифма правдоподобия. Произвольная функция
потерь не считается likelihood автоматически.

Обучение, sampling posterior и сжатие вызываются явно. Обычный `estimate` сам
не подбирает веса и не запускает MCMC. Для методов с прямым маршрутом он может
выполнить forward и рассчитать likelihood.

## 2. Основные объекты

Импорт: `from dl_complexity import ...`.

| Объект | Назначение | Основные параметры |
| --- | --- | --- |
| `ModelContext` | Модель и её вероятностное описание | `model`; необязательные `likelihood`, `loss`, `training` |
| `DataSource` | Источник батчей | `batches`, `n_observations=None`, `data_id=None` |
| `DataSource.from_tensors(x, y)` | Обернуть готовые тензоры одним батчем | `inputs`, `targets`, `data_id=None` |
| `TrainingRecipe` | Описание явного обучения | `train`, `objective`, `config={}` |
| `Provenance` | Идентификаторы входов | `state_id=None`, `data_id=None`, `likelihood_id=None` |
| `ComplexitySuite` | Вызвать несколько оценщиков | `methods`, `errors="raise"` |

Дополнительные поля `ModelContext`: `state_id`, `likelihood_id`,
`loss_likelihood_relation="unspecified"`, `runs`. Идентификаторы помогают обнаружить
противоречия между входами, но не являются контрольными суммами.

`context.structure` показывает модули и параметры без запуска модели.
`ModelContext.from_checkpoint(path, model_factory=..., map_location="cpu", **kwargs)`
загружает `state_dict` в заданную архитектуру; архитектуру нужно передать самому.

## 3. Общие методы check и estimate

Для оценщиков используется один интерфейс:

```python
report = method.check(context, data=data)
result = method.estimate(context, data=data)

# Альтернатива: передать готовый вход вместо вычислений по модели.
report = method.check(inputs=prepared)
result = method.estimate(inputs=prepared)
```

| Параметр | Что передавать |
| --- | --- |
| `context=None` | `ModelContext`, если метод поддерживает прямую оценку модели |
| `data=None` | `DataSource` для вычислений по данным |
| `inputs=None` | Подготовленную запись нужного типа; она заменяет расчёт входов по модели |

`check` возвращает `RequirementReport`: поля `status`, `missing`, `actions`,
`assumptions` и свойство `ready`. Статусы: `READY`, `MISSING_INPUTS`,
`PREPARATION_REQUIRED`, `UNSUPPORTED`. Готовность входов не доказывает, что модель
достигла MLE/MAP, posterior сошёлся или выполнены статистические предпосылки.

`estimate` возвращает `ComplexityResult`:

| Поле | Значение |
| --- | --- |
| `method` | Название метода |
| `kind` | Что именно оценено |
| `primary` | Основная величина; у диагностик может быть `None` |
| `quantities` | Именованные составляющие результата |
| `assumptions` | Условия и ограничения |
| `config` | Настройки расчёта |
| `provenance` | Идентификаторы входов |
| `elapsed_seconds` | Время оценки; отдельно выполненная подготовка сюда не входит |

Каждая `Quantity` хранит `value`, `unit`, `normalization`. Вызов
`quantity.to("bits")` переводит nats в bits, а `quantity.per_observation(n)`
делит допустимую суммарную величину на число объектов.
`native_ic`, кодовую длину и вероятность риска нельзя произвольно переводить друг в друга.

## 4. Пример: модель → BIC → результат

Из корня проекта установите библиотеку: `python -m pip install -e .`.
Следующий пример задаёт точное MLE нормальной модели, а не запускает обучение.

```python
import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.criteria import BIC
from dl_complexity.integrations.torch import GaussianLikelihood

x = torch.ones(4, 1, dtype=torch.float64)
y = torch.tensor([[1.], [2.], [4.], [5.]], dtype=torch.float64)
model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
with torch.no_grad():
    model.weight.fill_(3.)

context = ModelContext(model, likelihood=GaussianLikelihood(sigma=1.))
data = DataSource.from_tensors(x, y)
method = BIC()
assert method.check(context, data=data).ready
result = method.estimate(context, data=data)
print(result.primary.value.item())  # 18.737802626757272
print(result.primary.unit)          # native_ic
```

В `GaussianLikelihood(sigma=1.0)` параметр `sigma` — известное положительное
стандартное отклонение шума. Для классификации используют
`CategoricalLikelihood()` без параметров: модель возвращает logits, метки — индексы классов.

## 5. Информационные критерии

Импорт: `from dl_complexity.criteria import ...`.

| Метод / вход | Что делает | Параметры |
| --- | --- | --- |
| `AIC`, `BIC`, `HQIC` | Сочетают log likelihood и штраф за размер модели | `parameter_count="trainable"` |
| `WAIC()` | Критерий по pointwise likelihood при posterior-выборках | Настраиваемых параметров нет |
| `FitStatistics` | Вход AIC/BIC/HQIC | `log_likelihood`, `k`, `n=None`, `provenance=None` |
| `PosteriorLogLikelihood` | Вход WAIC | `log_likelihood` формы `[S, N]`, `provenance=None` |

`parameter_count`: `"trainable"` — обучаемые параметры, `"all"` — все параметры,
целое число — явно заданная статистическая размерность. При передаче `FitStatistics`
используется его `k`. `log_likelihood` — сумма натуральных логарифмов по данным,
не средняя loss. Для BIC/HQIC требуется `n > 1`.

Для WAIC нужны хотя бы две обычные posterior-выборки (`S >= 2`); один checkpoint
их не заменяет. Реализован WAIC2 с выборочной дисперсией `ddof=1`.
Основные значения критериев возвращаются в `native_ic`.

### Общая подготовка и несколько методов

`prepare_fit_statistics(context, data=..., parameter_count="trainable")`
вычисляет likelihood один раз. Пример продолжает предыдущий код:

```python
from dl_complexity import ComplexitySuite
from dl_complexity.criteria import AIC, HQIC
from dl_complexity.integrations.torch import prepare_fit_statistics

stats = prepare_fit_statistics(context, data=data)
suite = ComplexitySuite({"aic": AIC(), "bic": BIC(), "hqic": HQIC()})
outcomes = suite.estimate(inputs={name: stats for name in suite.methods})
for name, outcome in outcomes.items():
    print(name, outcome.result.primary.value.item())
```

У `ComplexitySuite` параметр `errors="raise"` означает остановку при ошибке,
`errors="collect"` — сохранение отдельных исходов с полями `result`,
`requirements`, `error`. `inputs` здесь — словарь по именам методов.
Suite не создаёт общий cache автоматически и не выбирает «лучший» метод.

## 6. Байесовские оценки

Импорт: `from dl_complexity.bayesian import ...`.

| Метод | Вход | Результат |
| --- | --- | --- |
| `WBIC()` | `TemperedLikelihood` | Средняя суммарная NLL при `beta=1/log(n)` |
| `VariationalCode()` | `VariationalTerms` | `expected_nll + kl_divergence`, отрицательный ELBO |
| `KFACLaplace()` | `LaplaceTerms` или `KFACFactors` | Приближение отрицательного log evidence |

У этих методов нет пользовательских настроек алгоритма в конструкторе;
существенные настройки задаются при подготовке входов. Результаты — в натах.

| Подготовленный вход | Поля |
| --- | --- |
| `TemperedLikelihood` | `log_likelihood`: вектор `[S]` суммарных log likelihood; `n`; `beta`; `provenance=None` |
| `VariationalTerms` | `expected_nll`, `kl_divergence`, `n=None`, `kl_estimator="analytic"`, `provenance=None` |
| `LaplaceTerms` | `log_likelihood`, `log_prior`, `dimension`, `logdet_precision`, `provenance=None` |
| `KFACFactors` | `log_likelihood`, `log_prior`, `blocks`, `prior_precision`, `curvature_kind`, `provenance=None` |
| `KroneckerBlock` | `activation_factor`, `output_factor`, `scale=1.0` |

`kl_estimator` принимает `"analytic"`, `"monte_carlo"`, `"supplied"`.
Likelihood и NLL суммируются по всей выборке; KL не масштабируется по числу батчей.
Для evidence через Laplace нужна точка MAP и согласованная полная precision.

Функции подготовки:

```text
prepare_variational_terms(q, prior, total_nll, *,
    n_samples=1, n=None, analytic_kl=True, provenance=None)

prepare_gaussian_linear_kfac(model, x, y, *,
    noise_std, prior_precision, provenance=None)
```

`q` и `prior` — распределения PyTorch, `total_nll(theta)` — суммарная NLL для
параметров `theta`, `n_samples` — число выборок для ожидания.
`noise_std` — стандартное отклонение Gaussian-шума, `prior_precision` — точность
изотропного Gaussian prior. Автоматический KFAC-preparer поддерживает один
Gaussian `nn.Linear`; факторы для произвольной сети передаются отдельно.

## 7. Posterior sampling для WAIC/WBIC

Импорт: `from dl_complexity.integrations.torch import ...`.

```text
Langevin(step_size, num_draws, burn_in=0, thin=1,
         metropolis=False, minibatch=False, seed=None)
```

| Параметр | Значение |
| --- | --- |
| `step_size` | Положительный шаг цепи |
| `num_draws` | Число сохраняемых выборок |
| `burn_in=0` | Начальные пропускаемые шаги |
| `thin=1` | Число шагов между сохранёнными выборками |
| `metropolis=False` | `True` включает MALA с поправкой Метрополиса |
| `minibatch=False` | `True` включает SGLD с одним батчем на шаг |
| `seed=None` | Seed; при `None` выбирается случайный |

MALA требует full batch: `metropolis=True` и `minibatch=True` несовместимы.

```text
sample_posterior(context, *, data, sampler, prior_precision,
                 beta=1.0, localization=0.0)
prepare_posterior_log_likelihood(context, *, data, sampler, prior_precision)
prepare_tempered_likelihood(context, *, data, sampler, prior_precision)
```

`sampler` — объект `Langevin`. `beta` задаёт степень likelihood;
`localization` — дополнительную локализацию около исходных весов, меняющую target.
Первая функция возвращает `PosteriorDraws`, две следующие — входы WAIC и WBIC.
WAIC-preparer использует обычный posterior, WBIC-preparer — `beta=1/log(n)`.
Число шагов само по себе не доказывает сходимость или хорошее смешивание цепи.

## 8. Моя ветка: prequential coding

Импорт: `from dl_complexity.coding import ...`.
Исходники: `src/dl_complexity/coding/prequential/`.

**Идея:** сначала оплатить метку по текущему прогнозу, затем обучиться на ней.
`prepare_*` выполняет обучение и записывает журнал. Оценщик проверяет и суммирует
готовый журнал; по последнему checkpoint восстановить эту историю нельзя.

| Функция | Протокол |
| --- | --- |
| `prepare_blockwise` | Новая модель на каждом прошлом префиксе |
| `prepare_online` | Одна модель, обновление на новом блоке после его оплаты |
| `prepare_replay` | Как online, но с повторением раскрытых примеров при обучении |

Общие параметры:

| Параметр | Смысл |
| --- | --- |
| `x` | Тензор `[N, ...]` с фиксированным порядком объектов |
| `y` | Метки `torch.long`, форма `[N]` |
| `n_classes` | Объявленное число классов, не меньше двух |
| `model_factory()` | Создать новое состояние модели, при необходимости с optimizer |
| `predict_logits=None` | Callback `(state, batch) → logits [B, C]`; без него состояние должно быть callable |
| `initial_code=None` | Суммарная стоимость начала **в натах**; по умолчанию равномерный код |
| `provenance=None` | Идентификаторы опыта |

### Blockwise

```text
prepare_blockwise(x, y, *, boundaries, model_factory, train, n_classes,
    predict_logits=None, initial_code=None,
    calibrator_factory=None, provenance=None)
```

- `boundaries`: возрастающие правые границы, например `[16,32,64,128,192]`.
  Первая задаёт равномерное начало, последняя должна равняться `N`.
- `train(state, prefix_x, prefix_y)`: обучить новую модель на прошлом префиксе.
- `calibrator_factory=None`: фабрика нового калибратора для каждого префикса.

### Online

```text
prepare_online(x, y, *, model_factory, update, n_classes, batch_size=1,
    predict_logits=None, initial_count=0, initial_code=None,
    calibrator=None, provenance=None)
```

- `update(state, new_x, new_y)`: обновить модель после оплаты нового блока.
- `batch_size=1`: размер нового кодируемого блока, не всей истории обучения.
- `initial_count=0`: число равномерно оплаченных начальных меток.
- `calibrator=None`: один калибратор на запуск, обновляемый после оплаты блока.

### Replay

```text
prepare_replay(x, y, *, model_factory, update, n_classes, replay_size,
    batch_size=1, predict_logits=None, initial_count=0, initial_code=None,
    calibrator=None, replay_policy=None, seed=0, provenance=None)
```

- `update(state, new_x, new_y, replay_x, replay_y)`: обучение с повторением примеров.
- `replay_size`: сколько раскрытых объектов выбирать; `0` отключает повторение.
- `replay_policy=None`: по умолчанию равномерная выборка без возвращения из
  раскрытого префикса; свой callback `replay_policy(stop)` возвращает индексы `< stop`.
- `seed=0`: seed выбора replay, а не инициализации модели или optimizer.

Replay выбирается после оплаты блока, поэтому может включать только что раскрытые
метки. Их стоимость второй раз не добавляется. Реализация использует доступ ко
всему префиксу, а не ограниченный reservoir buffer.

Callbacks могут менять состояние на месте и вернуть `None` либо вернуть новое
состояние. Они обязаны соблюдать причинность и не читать будущие метки из замыканий.

### Оценщики и журнал

`BlockwisePrequential(unit="nats")`, `OnlinePrequential(unit="nats")`,
`ReplayPrequential(unit="nats")` принимают соответствующий `PrequentialLedger`.
`unit` принимает `"nats"` или `"bits"`.

Журнал содержит `initial_nll`, `initial_count`, `entries`, `n`, `protocol`,
`provenance`. Запись `PrequentialEntry` хранит `start`, `stop`, `nll`,
`trained_until`, `calibrated_until=0`, `replay_indices=()`.
Границы — исключающие правые концы; NLL в журнале — суммарная, в натах.

Результат содержит `total`, `initial`, `predictive`, `per_observation`.

### Короткий работающий пример

```python
import math
import torch
from dl_complexity.coding import OnlinePrequential, prepare_online

x = torch.arange(4, dtype=torch.float64)[:, None]
y = torch.tensor([0, 1, 1, 0])

def predict(counts, batch):
    return counts.log().expand(len(batch), -1)

def update(counts, batch, labels):
    counts.add_(torch.bincount(labels, minlength=2))

ledger = prepare_online(
    x, y, model_factory=lambda: torch.ones(2, dtype=torch.float64),
    update=update, predict_logits=predict, n_classes=2,
)
result = OnlinePrequential(unit="bits").estimate(inputs=ledger)
assert math.isclose(result.primary.value.item(), math.log2(30), abs_tol=1e-12)
print(result.primary.value.item())  # 4.906890595608519
```

## 9. Температурная калибровка

`TemperatureScaler(temperature=1.0, steps=30, learning_rate=0.05)`:

| Параметр / метод | Значение |
| --- | --- |
| `temperature=1.0` | Положительная начальная температура T |
| `steps=30` | Число шагов Adam при каждом обновлении T |
| `learning_rate=0.05` | Шаг обучения температуры |
| `log_probs(logits)` | Рассчитать `log_softmax(logits / T)` |
| `update(logits, targets)` | Подогнать T по раскрытым меткам; `fit` — алиас |

Калибровка меняет уверенность, сохраняет argmax для фиксированных logits и
не обучает веса сети. Улучшение будущей NLL не гарантировано.
Для каждого независимого запуска нужен новый экземпляр калибратора.

## 10. Двухчастный MDL

Импорт: `from dl_complexity.coding import TwoPartMDL, prepare_two_part`.

```text
prepare_two_part(context, *, data, codec, model_factory=None,
    data_code_convention="discrete", resolution_adjustment_nats=0.0)
TwoPartMDL(unit="nats")
```

Функция кодирует и декодирует модель, затем оценивает likelihood **декодированной**
модели. `codec` задаёт бинарное описание; `model_factory` — необязательную фабрику
согласованной архитектуры. `unit` — `"nats"` или `"bits"`.

`data_code_convention="discrete"` применяется к дискретным меткам.
Для непрерывных плотностей нужно явно задать `"quantized"` и
`resolution_adjustment_nats` — поправку за разрешение. Плотность сама по себе
не является вероятностью отдельного наблюдаемого числа.

Можно передать готовый `TwoPartInputs(model_bits, data_nll, n, ...)`.
`model_bits` измеряется в битах, `data_nll` — в натах независимо от единиц выхода.
Результат содержит `total`, `model`, `data`, `per_observation`.

## 11. Сжатие и бинарный кодек

Импорт: `from dl_complexity.compression import ...`.

| Объект / функция | Параметры | Действие |
| --- | --- | --- |
| `MagnitudePruning` | `fraction=0.5`, от 0 до 1 | Обнулить малые matrix/kernel-веса; biases не затрагиваются |
| `LowRankApproximation` | `rank`, неотрицательное целое | Усечь SVD двумерных матриц весов |
| `UniformQuantization` | `bits=8`, от 1 до 16 | Округлить параметры на равномерную сетку |
| `uniform_quantize` | `tensor`, `bits=8` | Вернуть индексы уровней и метаданные; `.dequantize()` восстанавливает значения |
| `BinaryModelCodec` | `encoding="dense"`, `transform=None`, `bits=8`, `rank=1` | Создать бинарное сообщение модели |

Преобразования вызываются как `transform.apply(model)` и возвращают копию.
Доступные `encoding`: `"dense"`, `"sparse"`, `"low_rank"`, `"quantized"`.
`transform` позволяет применить преобразование перед кодированием.
`bits` используется для quantized, `rank` — для low-rank.

`codec.encode(model)` возвращает `EncodedModel` с `payload` и `length_bits`.
`codec.decode(encoded, model_factory=None)` восстанавливает модель.
Длина учитывает заголовок, архитектуру и метаданные. Одно лишь обнуление или
округление весов не гарантирует уменьшения размера сообщения.

## 12. Проекция Arora и граница риска

Проекция из `dl_complexity.compression`:

| Функция / класс | Параметры |
| --- | --- |
| `make_projection_helpers` | `model`, `n_projections`, обязательный `seed` |
| `matrix_project` | `matrix`, `helper`, `coefficient_alphabet` |
| `AroraMatrixProject` | `helpers`: словарь по именам параметров; `coefficient_alphabet`: конечная сетка |
| `nominal_projection_count` | `epsilon`, `eta`: настройки номинального числа проекций |

`AroraMatrixProject(...).project(model)` возвращает сжатое представление.
Helper-направления и алфавит должны быть согласованы с условиями конечного семейства.

Импорт границы: `from dl_complexity.bounds import ...`.

- `AroraCompressionBound(delta=0.05)` — `delta` задаёт вероятность нарушения границы.
- `prepare_arora_projection(original, projected, data, ...)` — сертификат для проекции.
- `prepare_arora_compression(original, compressed, data, ...)` — сертификат для
  явно заданной пары моделей и конечной параметризации.
- `arora_inputs_from_logits(original_logits, compressed_logits, targets, ...)` —
  подготовка из уже вычисленных logits всей выборки.

Обязательные настройки подготовки: `margin`, `helper_is_data_independent`,
`alphabet_is_data_independent`. В двух последних функциях дополнительно нужны
`parameter_count=q` и `values_per_parameter=r`; для проекции они берутся из её
конечного представления. Идентификатор `compressed_state_id=None` необязателен;
функция из logits также принимает `provenance=None`.

Вход `AroraCompressionInputs` хранит margin loss, размер выборки, `q`, `r`,
`margin`, максимальное искажение logits, декларации независимости и метаданные.
Сертификат должен охватывать всю обучающую выборку, искажение — не превышать
`margin/2`. Декларации независимости сами по себе её не доказывают.
Возвращается условная граница риска **сжатой** модели в `probability`.
Реализована Theorem 2.1; Theorem 4.1 не заявляется.

## 13. Диагностики и явное обучение

`WeightStatistics(trainable_only=False, spectral=False)` из
`dl_complexity.diagnostics` возвращает число параметров, L1/L2, долю нулей
и нормы отдельных тензоров. `trainable_only=True` ограничивает выборку обучаемыми
параметрами; `spectral=True` добавляет спектральные нормы двумерных матриц.
Вместо контекста можно передать `WeightInputs(weights, provenance=None)`.
Единого `primary` у диагностик нет.

`run_training(context, seed=...)` из `dl_complexity.integrations.torch` явно
запускает `context.training`: вызывает `train(model, **config)` с заданным внешним
seed, возвращает `TrainingRun` и записывает метаданные в `context.runs`.
Создание `TrainingRecipe` или вызов оценщика не запускают это обучение автоматически.

## 14. Где искать реализацию

| Папка / файл | Что находится |
| --- | --- |
| `src/dl_complexity/base.py`, `context.py`, `data.py`, `results.py` | Общий интерфейс и объекты |
| `src/dl_complexity/evaluation.py` | `ComplexitySuite` |
| `src/dl_complexity/criteria/` | AIC/BIC/HQIC/WAIC |
| `src/dl_complexity/bayesian/` | WBIC, variational, Laplace/KFAC |
| `src/dl_complexity/coding/prequential/` | Blockwise/online/replay и калибровка |
| `src/dl_complexity/coding/two_part.py` | Двухчастный MDL |
| `src/dl_complexity/compression/` | Преобразования, кодек и проекция Arora |
| `src/dl_complexity/bounds/` | Сертификаты и граница риска |
| `src/dl_complexity/diagnostics/` | Статистики весов |
| `src/dl_complexity/integrations/torch/` | Likelihood, sampling и обучение |

Короткое объяснение для выступления:

> Мы оборачиваем обычную модель PyTorch, задаём likelihood и данные, явно готовим
> нужные артефакты, проверяем входы и вызываем оценщик. Результат сохраняет своё
> математическое значение, единицы и ограничения. В моей ветке подготовка обучает
> модель только на раскрытых метках, записывает prequential-журнал, а оценщик
> рассчитывает по нему длину кода.
