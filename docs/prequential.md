# Последовательное кодирование: blockwise, online, replay

## Задача и реализованный объём

Prequential coding оценивает, сколько стоило предсказывать метки **по мере
обучения**, когда каждая новая метка ещё неизвестна. Быстро обучающийся предиктор
раньше назначает правильным ответам большие вероятности и тратит меньше битов.
Это характеристика модели **вместе с процедурой обучения и порядком данных**.

Ветка Курдюкова содержит три рабочих протокола, температурный калибратор,
проверяемый журнал и общий интерфейс оценки. Ниже приведены формулы, контракты
и исполняемые примеры; [демо MLP](demo.md) использует тот же публичный API.

Теоретическая основа:
[Bornschein, Li, Hutter, Sequential Learning of Neural Networks for Prequential MDL
(2022), §2 и Appendix B.1](https://arxiv.org/abs/2210.07931).
Реализованы chunk-incremental/from-scratch, online minibatch и uniform-prefix
replay. EMA, replay streams и аугментации из работы здесь не воспроизводятся.

## Формула и единицы

Пусть имеются упорядоченные пары $(x_i,y_i)$, $C$ классов и границы
$0 < b_0 < b_1 < \ldots < b_m=n$. Первый блок кодируется равномерно, остальные —
предиктором, который использовал только прошлые метки:

$$
L_{\mathrm{preq}} = b_0\ln C
  - \sum_{j=1}^{m}\sum_{i=b_{j-1}}^{b_j-1}
      \ln p_{\theta_j}(y_i\mid x_i).
$$

В online/replay допустим $b_0=0$: начальный код тогда равен нулю.
Внутри журнала все стоимости — **суммарные NLL в натах**, а не средняя
cross-entropy. Оценщик переводит результат в биты делением на $\ln 2$.

| Поле `result.quantities` | Значение |
|---|---|
| `total` | Начальный код + все последующие стоимости |
| `initial` | Стоимость начального префикса |
| `predictive` | Сумма стоимостей предсказанных блоков |
| `per_observation` | `total / n`, с нормировкой `per_observation` |

Это условный код **меток при известных входах**. Архитектура, инициализация,
порядок, алгоритм обучения и правила случайности должны быть общими для кодера
и декодера. Здесь вычисляется длина по вероятностям; бинарного арифметического
кодера меток и отдельного сообщения с правилами протокола нет.

## Причинный порядок

```text
Состояние, обученное только на прошлом
           |
      logits нового блока
           |
      текущая температура -> log probabilities -> записать NLL
                                                   |
                                           раскрыть метки
                                                   |
                                   обновить калибратор и модель
                                                   |
                                             следующий блок
```

Текущие метки передаются `update` только после фиксации стоимости. Калибровка
после раскрытия может изменить будущие вероятности, но не прошлую запись журнала.
Финальная сеть, оценённая на всей обучающей выборке, такой истории не даёт.

## Общие входы и callbacks

Импорт: `from dl_complexity.coding import ...`.

| Аргумент | Контракт |
|---|---|
| `x` | Материализованный Tensor `[n, ...]` с фиксированным порядком |
| `y` | `torch.long`, форма `[n]`, классы от `0` до `n_classes-1` |
| `n_classes` | Объявленный алфавит, целое число не меньше двух |
| `model_factory()` | Новое состояние: `nn.Module` либо объект с моделью и оптимизатором |
| `predict_logits(state, batch)` | Конечные logits `[len(batch), n_classes]`; необязателен для callable-состояния |
| `train` / `update` | Получает только разрешённые срезы; возвращает новое состояние либо `None` при изменении на месте |
| `provenance` | Необязательные объявленные идентификаторы в `Provenance` |

Если состояние — `nn.Module`, подготовка временно включает `eval` и `no_grad`
для прогноза и восстанавливает режимы модулей. Для собственного контейнера
режимы вложенной модели задаёт `predict_logits`. Оптимизатор нужно хранить
в состоянии, если его история должна сохраняться между online-обновлениями.

## Blockwise: обучение заново на каждом префиксе

`prepare_blockwise` получает `boundaries`, `model_factory` и
`train(state, prefix_x, prefix_y)`. Например, для `boundaries=[2,4,8]`:

| Шаг | Обучение | Кодирование |
|---|---|---|
| Начальный | Без обучения | `[0,2)` равномерным кодом |
| Первый прогноз | Новая модель на `[0,2)` | `[2,4)` |
| Второй прогноз | Ещё одна новая модель на `[0,4)` | `[4,8)` |

Границы — исключающие правые концы, последняя обязательно равна `n`.
По умолчанию первый блок стоит `boundaries[0] * ln(n_classes)` натов.
`initial_code` позволяет явно задать другую **суммарную** стоимость и требует
соответствующего общего правила декодирования.

Исполняемый пример с равномерным предиктором:

```{doctest}
>>> import torch
>>> from dl_complexity.coding import BlockwisePrequential, prepare_blockwise
>>> x = torch.arange(4, dtype=torch.float64)[:, None]
>>> y = torch.tensor([0, 1, 1, 0])
>>> ledger = prepare_blockwise(
...     x, y, boundaries=[1, 2, 4], model_factory=dict,
...     train=lambda state, prefix_x, prefix_y: None,
...     predict_logits=lambda state, batch: torch.zeros(len(batch), 2, dtype=torch.float64),
...     n_classes=2,
... )
>>> result = BlockwisePrequential(unit="bits").estimate(inputs=ledger)
>>> round(result.primary.value.item(), 6)
4.0
>>> [(entry.start, entry.stop, entry.trained_until) for entry in ledger.entries]
[(1, 2, 1), (2, 4, 2)]
```

Этот предиктор нужен для проверки арифметики. В демо `train` действительно
обучает MLP на каждом прошлом префиксе. Повторное обучение может быть дороже
online; кодовая длина сама по себе не учитывает секунды вычислений.

## Online: одно состояние и новые наблюдения

`prepare_online` создаёт одно состояние. Для каждого блока он записывает NLL,
затем вызывает `update(state, new_x, new_y)`. `batch_size` — размер **нового
кодируемого блока**, а не размер всей истории. Метки внутри блока становятся
доступны одновременно после его оценки.

При `initial_count > 0` сначала оплачивается начальный код, затем одним вызовом
`update` раскрывается весь начальный префикс. Начальный префикс не подгоняет
калибратор автоматически: первый следующий блок использует исходную температуру.

Аналитический пример: предиктор Bernoulli с prior Beta(1,1). Состояние хранит
два счётчика, `predict_logits` использует только прошлые значения:

```{doctest}
>>> import math
>>> from dl_complexity.coding import OnlinePrequential, prepare_online
>>> def factory():
...     return torch.ones(2, dtype=torch.float64)
>>> def predict(counts, batch):
...     return counts.log().expand(len(batch), -1)
>>> def update(counts, batch, labels):
...     counts.add_(torch.bincount(labels, minlength=2))
>>> ledger = prepare_online(
...     x, y, model_factory=factory, update=update,
...     predict_logits=predict, n_classes=2, batch_size=1,
... )
>>> code = OnlinePrequential(unit="bits").estimate(inputs=ledger)
>>> round(code.primary.value.item(), 6)
4.906891
>>> math.isclose(code.primary.value.item(), math.log2(30), abs_tol=1e-12)
True
```

Для последовательности `0,1,1,0` вероятности правильных меток равны
`1/2, 1/3, 1/2, 2/5`; их произведение — `1/30`. Это независимо известный
эталон, а не качество обученной сети.

## Replay: повторение уже раскрытых данных

`prepare_replay` использует callback
`update(state, new_x, new_y, replay_x, replay_y)`. После фиксации стоимости
блока `[start,stop)` выбирается до `replay_size` индексов из `[0,stop)`.
Текущий блок уже раскрыт и потому тоже может попасть в replay.

По умолчанию выборка равномерная без возвращения, с локальным генератором
и заданным `seed`. Полный короткий пример использует обучаемый предиктор
частот классов. Replay увеличивает счётчики повторённых меток и тем самым
меняет будущие вероятности; стоимость этих меток повторно не добавляется:

```{doctest}
>>> import torch
>>> from dl_complexity.coding import ReplayPrequential, prepare_replay
>>> replay_x = torch.arange(4, dtype=torch.float64)[:, None]
>>> replay_y = torch.tensor([0, 1, 1, 0])
>>> def replay_factory():
...     return torch.ones(2, dtype=torch.float64)
>>> def replay_predict(counts, batch):
...     return counts.log().expand(len(batch), -1)
>>> def replay_update(counts, new_x, new_y, past_x, past_y):
...     labels = torch.cat((new_y, past_y))
...     counts.add_(torch.bincount(labels, minlength=2))
>>> replay_ledger = prepare_replay(
...     replay_x, replay_y, model_factory=replay_factory, update=replay_update,
...     predict_logits=replay_predict, n_classes=2,
...     replay_size=2, batch_size=1, seed=9,
... )
>>> replay_code = ReplayPrequential(unit="bits").estimate(inputs=replay_ledger)
>>> bool(torch.isfinite(replay_code.primary.value))
True
>>> sum(entry.stop - entry.start for entry in replay_ledger.entries)
4
```

Это частотный learner с повторными обновлениями, а не точный Beta-posterior
по однократно учтённым наблюдениям из предыдущего примера. Полный вариант с
MLP, optimizer и конфигурацией доступен на [странице демо](demo.md).
`replay_size=0` отключает повторение; callback всё равно получает пустые
replay-тензоры. `replay_policy(stop)` может задать собственную политику,
возвращающую индексы раскрытого префикса. Будущие индексы отклоняются.

Повторённые объекты **не оплачиваются второй раз**: они изменяют состояние
для последующих прогнозов. Seed replay не фиксирует инициализацию модели,
оптимизатор или случайность внутри пользовательского callback.

Текущая реализация хранит весь `x/y`, журнал replay-индексов и строит
`torch.randperm(stop)` на каждом шаге. Это uniform-prefix replay, а не reservoir
buffer с памятью `O(replay_size)`. Стоимость перестановок может достигать
`O(n² / batch_size)`; размер replay-батча этого ограничения не устраняет.

## Температурная калибровка

`TemperatureScaler` преобразует logits $z$:

$$
\log p_T(y=c\mid x) = \operatorname{logsoftmax}(z/T)_c,\qquad T>0.
$$

`T=1` сохраняет обычные вероятности; `T>1` делает их менее уверенными.
`update` / `fit` оптимизирует `log(T)` по cross-entropy, используя отсоединённые
logits. Калибровка не обучает веса сети и не меняет argmax для фиксированных logits.
На каждом вызове создаётся новый Adam для одного скаляра.

| Протокол | Как подключить | На каких данных обновляется температура |
|---|---|---|
| Blockwise | `calibrator_factory=lambda: TemperatureScaler(...)` | Свежий калибратор на прошлом обучающем префиксе до следующего блока |
| Online / replay | `calibrator=TemperatureScaler(...)` | Текущий блок после записи его стоимости; действует на следующие блоки |

```{doctest}
>>> from dl_complexity.coding import TemperatureScaler
>>> scaler = TemperatureScaler(temperature=1.0)
>>> logits = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
>>> torch.allclose(scaler.log_probs(logits), torch.log_softmax(logits, dim=-1))
True
```

Для каждого независимого запуска создавайте новый калибратор. Повторное
использование подогнанного экземпляра может перенести метки из прошлого запуска.
В blockwise текущая реализация калибруется на том же прошлом префиксе, где
обучалась сеть; это причинно допустимо, но не независимая validation-калибровка.
На коротких или меняющихся блоках температура может ухудшить будущую NLL.
Пример и тесты не предполагают обязательного выигрыша от калибровки или replay.

## Журнал, проверки и воспроизводимость

`PrequentialLedger` содержит `initial_nll`, `initial_count`, `entries`, `n`,
`protocol` и `provenance`. Каждая `PrequentialEntry` записывает:

- `start`, `stop`: непрерывный кодируемый интервал;
- `nll`: его суммарную стоимость в натах;
- `trained_until`, `calibrated_until`: исключающие концы использованных префиксов;
- `replay_indices`: повторение **после** оценки этого интервала.

Оценщик проверяет полное однократное покрытие последовательности, подходящий
протокол, конечные неотрицательные стоимости, отсутствие будущих префиксов
и допустимость replay-индексов. Для пустого начального префикса требуется
нулевая стоимость. Готовый журнал можно повторно оценить без обучения. Здесь
используется `replay_ledger` из примера выше и соответствующий ему оценщик:

```{doctest}
>>> method = ReplayPrequential(unit="bits")
>>> method.check(inputs=replay_ledger).ready
True
>>> replay_code = method.estimate(inputs=replay_ledger)
>>> replay_code.quantities["per_observation"].normalization
'per_observation'
```

Для сохранения опыта фиксируйте данные и порядок, seed инициализации, архитектуру,
границы, обучение, optimizer, replay, калибровку и время **всей подготовки**.
Демо сохраняет это вместе с журналом в JSON. Сам `PrequentialLedger` не сохраняет
optimizer, checkpoint и конфигурацию всех callbacks за пользователя.

## Проверки и ограничения

| Проверка | Где находится |
|---|---|
| Predict-before-update, prefix-only обучение, границы и replay | `tests/test_coding.py` |
| Точный Beta–Bernoulli online-код и фиксированные blockwise вероятности | `tests/test_analytical_codes_bounds.py` |
| MLP, повторяемость, сумма стоимостей, replay без повторной оплаты | `tests/test_prequential_demo.py` |
| Исполняемые короткие примеры этой страницы | Sphinx `doctest` builder |

Подготовка поддерживает конечную категориальную задачу на материализованных
тензорах. Автоматического протокола для регрессии, распределённого обучения,
контроля скрытых замыканий callback и полноценного бинарного кодера нет.
Проверка метаданных не доказывает причинность произвольного Python-кода.
Полные публичные сигнатуры: [API последовательных кодов](api/coding.rst).
