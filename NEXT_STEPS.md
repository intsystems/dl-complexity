# Next steps

## Что уже сделано

- [x] README заменён описанием проекта *Theoretically informed deep learning model
  complexity estimation*.
- [x] Участники, проектные роли и алгоритмические ветки распределены.
- [x] Описаны исследовательский вопрос, верхнеуровневая архитектура и стек.
- [x] Выбраны имена: **DL Complexity** для проекта и `dl_complexity` для Python-пакета.
- [x] Подготовлен план методов: AIC/BIC/HQIC, WAIC/WBIC, 2-part MDL, variational
  coding, KFAC-Laplace evidence, prequential coding и compression-based generalization.
- [x] Созданы индекс литературы и презентация для Tech meeting 1.

## Ближайшие задачи

1. Зафиксировать публичные типы `ComplexityEstimator`, `ComplexityResult` и
   `BenchmarkConfig`: единицы, входные артефакты, метаданные и обработку ошибок.
2. Создать чистый пакет `src/dl_complexity/` без кода шаблонного `mylib`.
3. Сделать PoC на небольшом MLP: BIC, 2-part MDL и blockwise prequential code на одном
   фиксированном split.
4. Настроить установку через `pyproject.toml`, `pytest`, coverage и CI.
5. Добавить синтетический benchmark и MNIST/Fashion-MNIST; сохранять seed, split,
   конфигурацию обучения, время и память.
6. Подготовить промежуточную документацию, демо и черновики блог-поста/техотчёта к
   Tech meeting 2.

## Критерий завершения проекта

- все estimator-ы работают через один интерфейс и возвращают nats/bits на объект;
- методы проверены unit- и integration-тестами, coverage > 90%;
- общий benchmark сравнивает score с test NLL и generalization gap;
- установка, документация и демо воспроизводимы с чистого окружения;
- каждый алгоритм прошёл cross-review.

Работа с кодом: отдельная ветка → pull request в `master` → review.
