# Презентации DL Complexity

## Архитектура библиотеки и BIC proof of concept

Основная презентация — **6 слайдов на английском** о существующем API:

- [PDF](proof-of-concept/main.pdf);
- [LaTeX-исходник](proof-of-concept/main.tex);
- [оформление](proof-of-concept/theme.tex).

Содержание: назначение библиотеки, объекты и поток данных, базовые классы,
пользовательские вызовы, реализация BIC и сравнение с аналитическим эталоном.
Схемы редактируются в TikZ, примеры кода — обычный текст LaTeX.

### Сборка

Нужны XeLaTeX, latexmk и шрифт Arial. Из корня репозитория:

```bash
cd presentation/proof-of-concept
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

Промежуточные файлы остаются в игнорируемой папке `build/`.

### Пример со слайдов 4–6

Для нормальной модели с известной дисперсией `sigma² = 1` и наблюдениями
`y = (1, 2, 4, 5)` оценка максимального правдоподобия равна `mu = 3`.
Модель `Linear(1, 1, bias=False)` на единичных входах представляет этот общий
параметр среднего. Вес задаётся аналитически: скрытого обучения нет.

Запустить в окружении с доступным `dl_complexity`:

```python
import math

import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.criteria import BIC, FitStatistics
from dl_complexity.integrations.torch import GaussianLikelihood

x = torch.ones(4, 1, dtype=torch.float64)
y = torch.tensor([1., 2., 4., 5.], dtype=torch.float64)[:, None]
model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
with torch.no_grad():
    model.weight.fill_(3.)

context = ModelContext(model, likelihood=GaussianLikelihood(sigma=1.))
data = DataSource.from_tensors(x, y)
bic = BIC()
report = bic.check(context, data=data)
assert report.ready, report
result = bic.estimate(context, data=data)

# Independent reference: n=4, k=1, squared residuals sum to 10.
ll = -2 * math.log(2 * math.pi) - 5
stats = FitStatistics(log_likelihood=ll, k=1, n=4)
prepared = bic.estimate(inputs=stats)
reference = {
    "deviance": -2 * ll,
    "penalty": math.log(4),
    "bic": -2 * ll + math.log(4),
}
for answer in (result, prepared):
    for name, expected in reference.items():
        quantity = answer.quantities[name]
        assert quantity.unit == "native_ic"
        assert quantity.normalization == "total"
        assert math.isclose(
            quantity.value.item(), expected, rel_tol=0, abs_tol=1e-12
        )
print(result.primary.value.item())  # 18.737802626757272
```

`check` проверяет готовность входов, а не оптимальность параметров. Здесь MLE
известно из аналитического решения. BIC сохраняет собственную шкалу `native_ic`:
это не точная кодовая длина и не граница риска.

Пять вариантов нормальной модели с известной дисперсией проверяются в
[`test_analytical_criteria.py`](../tests/test_analytical_criteria.py):

```bash
python -m pytest tests/test_analytical_criteria.py -q -k gaussian_mean_mle_known_variance
```

## Архив: Tech Meeting 1

Предыдущие файлы сохранены без изменения содержимого:

- [PDF](archive/tech-meeting-1/dl_complexity_presentation.pdf);
- [LaTeX-исходник](archive/tech-meeting-1/dl_complexity_presentation.tex);
- [исходный README](archive/tech-meeting-1/README.md).

Архив описывает прежний план проекта, а не текущие публичные интерфейсы.
Его README сохранён как исторический документ; ссылка на заметки докладчика
внутри него не имеет соответствующего файла.
