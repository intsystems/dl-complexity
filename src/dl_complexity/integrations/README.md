# Интеграции

_Адаптеры внешних вычислительных интерфейсов к объектам DL Complexity._

---

## 🔌 PyTorch

[Модуль torch](torch/README.md) содержит likelihood для классификации и
гауссовской регрессии, подготовку `FitStatistics` и явный запуск
`TrainingRecipe`. Его публичные объекты импортируются из
`dl_complexity.integrations.torch`.

Импорт `dl_complexity.integrations` не запускает обучение и не собирает
активации или градиенты. Других backend-адаптеров в пакете нет.
