# Установка и сборка документации

## Окружение

Нужны Python 3.11+ и PyTorch 2.2+. GPU, внешние датасеты и доступ к сети
после установки зависимостей для демонстрации prequential coding не нужны.
Из корня репозитория:

```bash
python -m venv .venv
```

Активация в PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Активация в Linux/macOS:

```bash
source .venv/bin/activate
```

Установка библиотеки для разработки:

```bash
python -m pip install -e ".[test,docs]"
```

Для использования без инструментов разработки достаточно `python -m pip install .`.
Установка из исходников описана здесь; публикация пакета в PyPI не предполагается.

## Проверка

```bash
python examples/quickstart.py
python examples/prequential_demo.py
python -m pytest -q
```

Тесты только ветки последовательных кодов и её интеграционного примера:

```bash
python -m pytest tests/test_coding.py tests/test_prequential_demo.py -q
python -m pytest tests/test_analytical_codes_bounds.py -q -k "online or blockwise or replay or temperature"
```

## HTML-документация

Используются Sphinx, тема Read the Docs и MyST для Markdown. Публичные
сигнатуры и описания API берутся непосредственно из Python-модулей.
Из корня проекта:

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m sphinx -b doctest -W docs docs/_build/doctest
```

Откройте `docs/_build/html/index.html` обычным браузером. Сайт содержит
навигацию, поиск, формулы и ссылки на исходники. Для отображения формул
браузер загружает MathJax; сама сборка не скачивает датасеты и не обучает сеть.
`-W` делает предупреждения сборки ошибками. Промежуточные файлы игнорируются Git.

Workflow `.github/workflows/docs.yml` проверяет сборку, исполняемые примеры
документации и тесты prequential-ветки, затем сохраняет HTML как артефакт CI.
Настройка публикации GitHub Pages остаётся отдельным действием владельца проекта.
