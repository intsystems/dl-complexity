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

Для [демонстрационного ноутбука](demo.md) предусмотрены отдельные зависимости;
они не нужны обычному пользователю библиотеки:

```bash
python -m pip install -e ".[demo]"
python -m jupyterlab examples/demo.ipynb
```

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
документации, полный запуск демонстрационного ноутбука и тесты prequential-ветки,
затем сохраняет HTML как артефакт CI.

## Публикация сайта через GitHub Pages

Workflow подготовлен для **ручной публикации из `master`**. Push и pull request
проверяют документацию; обновление публичного сайта запускается отдельно.
Готовый HTML из `docs/_build/` добавлять в Git не нужно.

1. Закоммитьте исходники документации и изменения
   `.github/workflows/docs.yml`, отправьте свою ветку в GitHub и влейте pull request
   в `master`. Новые файлы, например `docs/library_guide.md`, тоже должны войти в коммит.
2. В репозитории откройте **Settings → Pages → Build and deployment → Source**
   и выберите **GitHub Actions**. Если нет доступа к Settings, этот шаг выполняет
   администратор репозитория.
3. Откройте **Actions → Documentation and prequential checks → Run workflow**,
   выберите **master** и запустите workflow.
4. Дождитесь успешного завершения `docs` и `deploy`. Ссылка на опубликованный сайт
   появится в окружении `github-pages` и в Settings → Pages. Если окружение требует
   подтверждения deployment, его должен выполнить назначенный reviewer.

Стандартный адрес для этого репозитория:
`https://intsystems.github.io/dl-complexity/`. Он станет доступен после первого
успешного deployment. Последующие изменения публикуются тем же ручным запуском.
Кнопка Run workflow доступна после появления workflow в основной ветке;
для запуска нужны права записи в репозиторий.

Без публикации сайт можно скачать как ZIP-артефакт
`dl-complexity-documentation` из успешного workflow, распаковать и открыть
`index.html`. Для скачивания артефактов GitHub требуется вход в аккаунт.

Настройка соответствует официальной документации GitHub:
[источник публикации](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site),
[workflow Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[ручной запуск](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
