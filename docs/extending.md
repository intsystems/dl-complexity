# Расширение библиотеки

## Собственный learner для prequential

Новый optimizer или архитектура обычно не требуют нового оценщика. Реализуйте
`model_factory`, `predict_logits` и callback обучения для существующего протокола.
Состояние может содержать модель и optimizer, как `Learner` в
[демо с полным исходником](demo.md).

```python
def model_factory():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return {"model": model, "optimizer": optimizer}

def predict_logits(state, batch):
    state["model"].eval()
    with torch.no_grad():
        return state["model"](batch)

def update(state, new_x, new_y):
    state["model"].train()
    optimizer = state["optimizer"]
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(state["model"](new_x), new_y)
    loss.backward()
    optimizer.step()
```

Эти callbacks подключаются к `prepare_online`. Для replay добавьте два аргумента
и явно определите, как новый и повторённый батчи участвуют в loss. Для blockwise
фабрика создаёт независимый learner для каждого прошлого префикса.

Собственный callback должен соблюдать причинность: не читать будущие метки из
замыкания, глобального массива или checkpoint, обученного на всей выборке.
Новая replay-политика получает `stop` и возвращает только индексы `< stop`.
Изменение буфера или закона выборки нужно отразить в конфигурации опыта.

## Собственный калибратор

Протокол ожидает два метода:

```python
class Calibrator:
    def log_probs(self, logits):
        # Возвращает finite normalized log probabilities той же формы.
        ...

    def update(self, logits, targets):
        # Только раскрытые метки, после записи NLL текущего блока.
        ...
```

`log_probs` не должен использовать метки кодируемого блока. Выход проверяется
на конечность и `logsumexp(..., dim=-1) ≈ 0`. Для независимых опытов нужны
независимые экземпляры; blockwise использует `calibrator_factory`.

## Новый математический оценщик

Если требуется другая величина, создайте типизированный вход и наследника
`ComplexityEstimator` либо семейной базы:

```{doctest}
>>> from dataclasses import dataclass
>>> from dl_complexity import ComplexityEstimator, ComplexityResult, Quantity
>>> @dataclass(frozen=True)
... class FixedCode:
...     bits: float
>>> class FixedCodeEstimator(ComplexityEstimator):
...     input_type = FixedCode
...     uses_data = False
...     def _estimate(self, inputs):
...         if inputs.bits < 0:
...             raise ValueError("Code length must be nonnegative")
...         return ComplexityResult(
...             method="FixedCode", kind="declared_code_length",
...             quantities={"total": Quantity(inputs.bits, "bits")},
...             primary_name="total",
...         )
>>> FixedCodeEstimator().estimate(inputs=FixedCode(8.0)).primary.value.item()
8.0
```

Это минимальный пример механики расширения. Для публичного метода добавьте
проверки типа/формы в `_check_inputs`, численную валидацию при вычислении,
`assumptions`, provenance и необходимые настройки в результат.
Прямой маршрут из контекста добавляется через `_check_context` и
`_prepare_context`, только когда не требует скрытого обучения.

Завершение новой реализации включает публичный экспорт в `__init__.py`,
страницу метода и API, исполняемый пример и проверку по независимо известной
формуле. Не ослабляйте проверки существующего журнала ради другого протокола:
новой схеме доступа к меткам может понадобиться собственный тип входа.
