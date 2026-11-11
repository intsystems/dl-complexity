# DL Complexity

Библиотека теоретически обоснованных оценок сложности моделей PyTorch.
Один интерфейс объединяет информационные критерии, байесовские оценки,
кодовые длины и компрессионную границу риска. Каждый результат сохраняет
свои единицы, составляющие и допущения.

Исследовательский вопрос проекта: как оценки сложности связаны с test NLL,
generalization gap и устойчивостью ранжирования при изменении архитектуры,
размера выборки и регуляризации?

Начните с [установки](installation.md), затем запустите
[демонстрацию](demo.md). Для последовательного кодирования есть отдельное
[руководство с формулами и рабочими примерами](prequential.md).

```{toctree}
:maxdepth: 2
:caption: Начало работы

installation
demo
```

```{toctree}
:maxdepth: 2
:caption: Библиотека и методы

architecture
methods
prequential
experiments
extending
```

```{toctree}
:maxdepth: 2
:caption: Справочник

api/index
```

## Состояние проекта

Реализованы AIC/BIC/HQIC, WAIC2, WBIC, variational free energy, Laplace/KFAC,
двухчастный MDL, три prequential-протокола, калибровка, преобразования весов,
бинарный кодек и условная граница Arora. Набор примеров работает на CPU.
Общий benchmark на MNIST/Fashion-MNIST и итоговое исследование ещё предстоят;
демонстрационные числа не заменяют это исследование.

## Материалы проекта

- [Исходный код и README](https://github.com/intsystems/dl-complexity).
- [Подробный разбор существующих вычислений](https://github.com/intsystems/dl-complexity/blob/master/docs/CODE_WALKTHROUGH.md).
- [Аналитические эталоны и тесты](https://github.com/intsystems/dl-complexity/blob/master/tests/README.md).
- [Пять первоисточников и локальные PDF](https://github.com/intsystems/dl-complexity/blob/master/literature/README.md).
- [Презентация архитектуры и BIC PoC](https://github.com/intsystems/dl-complexity/blob/master/presentation/README.md).

Организация сайта следует примеру
[Epiplexity](https://github.com/intsystems/Epiplexity/tree/main/doc/source):
установка, архитектура, методы, демонстрация, эксперименты, расширение и API
из docstrings. Описание алгоритмов и примеры соответствуют `dl_complexity`.
