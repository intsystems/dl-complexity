# DL Complexity — подробные заметки докладчика

Эти заметки написаны на русском для англоязычной презентации из 16 слайдов. Текст не нужно читать дословно: основная часть в каждом разделе — рекомендуемый рассказ, а определения и вопросы предназначены для подготовки к обсуждению.

## Стратегия выступления на 10–15 минут

Цель доклада — за 13–14,5 минуты последовательно ответить на пять вопросов:

1. Какую проблему решает проект?
2. Что именно будет результатом проекта?
3. Какие семейства оценок мы реализуем и чем они отличаются?
4. Как мы обеспечим честное экспериментальное сравнение?
5. Кто и в какой последовательности это реализует?

Рекомендуемый базовый тайминг — **13–14,5 минуты** в зависимости от темпа и числа
коротких пояснений:

| Слайды | Содержание | Время |
|---|---|---:|
| 1–3 | вводная часть, вопрос и границы проекта | 1:50 |
| 4–5 | интерфейс и архитектура | 1:35 |
| 6–9 | четыре алгоритмических блока | 5:00 |
| 10–11 | benchmark и демонстрация | 1:40 |
| 12–14 | роли, этапы и риски | 1:50 |
| 15–16 | вывод и литература | 1:05 |

В таблице около 13 минут основного рассказа; при лимите 15 минут остаётся до
полутора минут на переключение слайдов, короткие паузы и одно-два уточнения.

Если дано только 10 минут, сократите слайды 3, 5, 12–14 до одного тезиса каждый и не проговаривайте литературу по пунктам. Если есть 15 минут, используйте дополнительное время на слайды 6–10: именно там преподаватель увидит, что проект технически осмыслен.

Общая линия рассказа:

> Количество параметров недостаточно описывает эффективную сложность нейросети. Мы создаём библиотеку, которая реализует несколько теоретически мотивированных приближений сложности через единый честный интерфейс, а затем проверяем, какие из них действительно связаны с качеством обобщения при контролируемых изменениях данных, архитектуры и регуляризации.

Сохраняйте аккуратность формулировок:

- не говорите, что мы измеряем единственную «истинную сложность»;
- не обещайте, что корреляция доказывает причинность или гарантирует обобщение;
- не называйте AIC/BIC строгими нейросетевыми bounds;
- различайте **значение критерия**, **длину кода**, **оценку evidence** и **generalization bound**;
- при переводе между шкалами указывайте основание логарифма: натуральный логарифм даёт nats, логарифм по основанию 2 — bits.

---

## Слайд 1. DL Complexity

**Рекомендуемое время:** 20–25 секунд.

### Что сказать

«Наш проект называется DL Complexity, а имя Python-пакета — `dl_complexity`. Полное название — *Theoretically informed deep learning model complexity estimation*. Мы хотим реализовать и сопоставить несколько теоретически мотивированных способов оценивать сложность обученной нейросети. Сначала я сформулирую исследовательский вопрос, затем покажу интерфейс и архитектуру, кратко объясню четыре группы алгоритмов и закончу экспериментальным протоколом и распределением работ.»

### Термины

**Theoretically informed** означает, что оценки связаны с информационными критериями, байесовским выводом, принципом минимальной длины описания, последовательным кодированием или теорией обобщения. Это не означает, что все оценки точны или что их предпосылки выполняются для глубоких сетей.

**Model complexity** здесь — не одно универсальное число. В зависимости от метода это штраф за число параметров, длина описания параметров или предсказаний, KL-расхождение с prior, curvature-dependent Occam factor либо член в границе обобщения.

### Связка

«Начнём с того, почему обычного количества параметров для такой задачи недостаточно.»

### Вероятные вопросы

**Почему название проекта не содержит MDL?**  
Scope шире строгого MDL: AIC/BIC, WAIC, Laplace evidence и compression bounds имеют разные исходные постановки. Кодовая интерпретация объединяет многие из них, но не делает их тождественными.

**Вы оцениваете архитектуру или конкретно обученную модель?**  
В основном конкретную обученную модель вместе с данными, likelihood, prior и процедурой обучения. Некоторые оценки зависят преимущественно от архитектуры, другие — также от найденных весов и learner-а.

---

## Слайд 2. Problem statement

**Рекомендуемое время:** 50–60 секунд.

### Что сказать

«Число параметров — полезный, но слишком грубый baseline. Две сети с одинаковым числом параметров могут представлять функции разной эффективной сложности из-за структуры весов, регуляризации, posterior и процедуры обучения. И наоборот, сильно переопределённая сеть может хорошо обобщать.

Наш исследовательский вопрос: какие теоретически мотивированные оценки сложности лучше согласуются с test negative log-likelihood и generalization gap, когда мы меняем размер данных, архитектуру и регуляризацию?

Мы реализуем несколько семейств оценок через общий интерфейс, сравним их в одном протоколе и, кроме score, измерим стоимость вычисления, стабильность ранжирования и выполнимость предпосылок. Мы не предполагаем существование универсальной истинной меры сложности: предмет исследования — поведение разных приближений.»

### Термины и формулы

**Переопределённая модель** имеет больше настраиваемых параметров, чем подсказывает размер задачи, иногда больше числа обучающих объектов. Это не тождественно переобучению: overparameterization описывает параметризацию, overfitting — ухудшение качества на новых данных.

**Likelihood** — вероятность наблюдаемых данных при фиксированных параметрах: $p(D\mid\theta)$. Для классификации отрицательный логарифм likelihood совпадает с суммарной cross-entropy при categorical model.

**Test NLL**:

\[
\operatorname{NLL}_{\mathrm{test}}
=-\frac{1}{n_{\mathrm{test}}}\sum_{i=1}^{n_{\mathrm{test}}}
\log p_{\theta}(y_i\mid x_i).
\]

Чем меньше NLL, тем большую вероятность модель в среднем присваивает правильным ответам. В отличие от accuracy, NLL учитывает уверенность и чувствителен к калибровке.

**Generalization gap** в базовом протоколе:

\[
\operatorname{gap}_{\mathrm{NLL}}
=\operatorname{NLL}_{\text{test}}-\operatorname{NLL}_{\text{train}}.
\]

Можно дополнительно использовать разность ошибок классификации, но определения нельзя смешивать. Малый gap сам по себе не означает хорошую модель: она может одинаково плохо работать на train и test. Поэтому gap анализируется вместе с test NLL.

**Стабильность ранжирования** — насколько порядок моделей по complexity score сохраняется при смене seed, split или размера выборки.

### Связка

«Чтобы этот вопрос оставался выполнимым, мы фиксируем конкретный результат и явно ограничиваем первую версию проекта.»

### Вероятные вопросы

**Почему test NLL, а не accuracy?**  
NLL является proper scoring rule, различает одинаково точные, но по-разному калиброванные модели и естественно связана с likelihood, evidence и длиной кода. Accuracy останется вторичной метрикой.

**Не смешивается ли complexity с качеством fit?**  
Многие критерии намеренно состоят из fit term и complexity penalty. Поэтому компоненты сохраняются отдельно; мы анализируем полный критерий и его штрафную часть, а train/validation NLL включаем как baselines.

**Что значит «лучше согласуется»?**  
В первую очередь более сильная ранговая корреляция Spearman в ожидаемом направлении, затем стабильность результата по seeds и sample sizes.

**Корреляция доказывает, что мера объясняет обобщение?**  
Нет. Это эмпирическая диагностическая связь, не причинный вывод и не доказательство bound.

---

## Слайд 3. Project deliverables and scope

**Рекомендуемое время:** 40–45 секунд.

### Что сказать

«Результатом будет устанавливаемая Python-библиотека с единым benchmark runner, воспроизводимыми конфигурациями, таблицами, графиками, документацией и demo. В первой версии мы ограничиваемся supervised classification, небольшими MLP и CNN, синтетическими данными, MNIST и Fashion-MNIST и PyTorch-моделями.

Мы сознательно не включаем большие language models, distributed training, автоматический выбор лучшей модели и доказательство новых bounds. Сейчас проект находится на стадии планирования: этот слайд описывает целевой результат, а не уже готовую функциональность.»

### Термины

**Supervised classification** — обучение по парам $(x_i,y_i)$, где $y_i\in\{1,\ldots,K\}$.

**MLP** — multilayer perceptron, полносвязная сеть. **CNN** — convolutional neural network. Малые модели выбраны потому, что некоторые оценки требуют posterior sampling, curvature computation или многократного переобучения.

**Checkpoint-based evaluation where possible** означает, что дешёвые post-hoc методы
получают обученную модель и сохранённые артефакты. Это не общее ограничение API:
WAIC/WBIC и variational методы могут выполнять posterior inference, Laplace — оценивать
curvature, compression — строить новое дискретное представление, а prequential estimator —
запускать обучение на последовательных подвыборках. Для этого context предоставляет
posterior samples, curvature/compression backends или training callback.

Воспроизводимая конфигурация хранит версию данных, индексы split, seed, архитектуру, optimizer, stopping rule, likelihood, prior и параметры estimator-а.

### Связка

«Дальше покажу, как разные по смыслу методы приводятся к общему программному контракту без потери их исходных шкал и предпосылок.»

### Вероятные вопросы

**Почему MNIST и Fashion-MNIST, если задача про deep learning?**  
Это контролируемая площадка для проверки корректности и стоимости методов. Некоторые из них вычислительно дороги. Архитектура пакета при этом не привязана к этим наборам.

**Почему не CIFAR-10?**  
Его можно добавить после end-to-end PoC. На первой итерации prequential coding требует нескольких обучений, KFAC/Laplace — curvature computation, поэтому ограничение масштаба оправдано.

**Что значит «устанавливаемая библиотека»?**  
Пакет устанавливается из чистого окружения, импортируется как `dl_complexity`, имеет объявленные зависимости и воспроизводимый пример запуска.

---

## Слайд 4. Project name and common interface

**Рекомендуемое время:** 50–55 секунд.

### Что сказать

«Общий интерфейс — `ComplexityEstimator.estimate(context) -> ComplexityResult`. Context содержит модель, dataset и split, likelihood, training artifacts и seed. Result хранит исходное значение метода, допустимые переводы в nats или bits, нормировку per sample, компоненты, предпосылки, warnings, runtime, peak memory и metadata. Config содержит prior, precision, block schedule, число posterior samples и параметры compression.

Мы не скрываем различия шкал. Например, AIC, BIC и HQIC определены в шкале
$-2\log \widehat L$, где $\widehat L=p(D\mid\widehat\theta)$ — максимизированный
likelihood. При натуральном логарифме деление критерия на два даёт NLL-like величину
в nats, но не превращает критерий в фактический префиксный код. Поэтому native и
converted values хранятся отдельно.»

### Подробное объяснение API

**`ComplexityEstimator`** — общий протокол или абстрактный класс. Реализации: `BICEstimator`, `VariationalCodeEstimator`, `PrequentialEstimator` и другие.

**`EstimationContext`** отделяет общие входы от конфигурации метода:

- `model` — обученный `torch.nn.Module`;
- `dataset` и `split` — данные и точные индексы разбиения;
- `likelihood` — функция pointwise $\log p(y_i\mid x_i,\theta)$;
- `training_artifacts` — checkpoint, optimizer state, posterior samples, curvature factors или training callback;
- `seed` — источник детерминированности.

**`ComplexityResult`** содержит native value, units, допустимые nats/bits, per-sample value, компоненты fit/penalty, assumptions, warnings, runtime, peak memory и metadata.

Для information content используем отдельное обозначение
$I=-\ln p$: единица — nat. В bits та же информация равна
$-\log_2p=-\ln p/\ln2$. Полные значения AIC/BIC/HQIC/WAIC заданы в deviance
scale: у классических criteria fit term основан на $\widehat L$, а у WAIC — на
posterior-predictive lppd. Для NLL-like nats делим весь criterion на 2, а для bits
затем ещё на $\ln2$. Это не превращает результат в автоматически реализованную
code length.

Нормировка per sample полезна для разных $n$, но суммарное значение сохраняется, потому что штрафы имеют суммарный смысл.

### Связка

«Этот контракт определяет границу между входными adapters, конкретными estimator-ами и общим benchmark runner.»

### Вероятные вопросы

**Почему нельзя вернуть одно число `float`?**  
Одинаковое число может иметь разный смысл и единицы. Без компонентов, assumptions и metadata результат нельзя корректно сравнить или воспроизвести.

**Можно ли напрямую сравнить BIC в nats и prequential code в bits?**  
Нет: даже после перевода обоих значений в bits одинаковые единицы не создают общей
семантики. BIC — асимптотическая evidence approximation с собственными additive
constants и assumptions; prequential length — операционная длина, зависящая от learner,
ordering и block schedule. Их можно выводить рядом и сравнивать predictive utility или
ranks внутри заранее объявленного target, но нельзя считать абсолютные значения
взаимозаменяемыми.

**Зачем warnings в результате?**  
Предпосылки зависят от запуска: WAIC нельзя корректно вычислить без posterior samples, Laplace score может быть нестабилен при плохом damping.

**Что является likelihood для softmax-сети?**  
Categorical likelihood: softmax probability истинного класса. Суммарная NLL совпадает с cross-entropy при `reduction="sum"`; при `mean` нужно умножить на число объектов.

---

## Слайд 5. Library architecture, integration, and technology stack

**Рекомендуемое время:** 45–50 секунд.

### Что сказать

«Поток данных идёт от adapters для модели и данных через registry и validation к одному из estimator-ов, после чего получается структурированный result и report. Над этим расположен `BenchmarkRunner`: он обучает или загружает модели, запускает оценки, агрегирует результаты и строит таблицы и графики.

Основная интеграция — с `torch.nn.Module` и `DataLoader`. Отдельные adapters задают likelihood, posterior samples и при необходимости curvature backend для KFAC/Laplace. Результат сериализуется в JSON или CSV. Базовый стек — Python, PyTorch, NumPy/SciPy, pandas, scikit-learn, pytest и инструменты документации и demo.»

### Термины и архитектурные решения

**Adapter** преобразует внешний объект в ожидаемый библиотекой протокол. Это позволяет не встраивать в каждый estimator код для каждой модели или posterior library.

**Registry** сопоставляет строковое имя, например `"bic"`, с классом estimator-а. Он нужен YAML/JSON-конфигурациям и CLI; при этом пользователь может создать estimator напрямую.

**Validation** проверяет вход до дорогого вычисления: есть ли pointwise likelihood, posterior samples, training callback, согласованы ли units и split.

**Curvature backend** вычисляет Hessian, generalized Gauss–Newton, Fisher или их приближения. Он отделён, потому что Laplace estimator не должен зависеть от одной реализации KFAC.

**JSON/CSV-compatible result.** В metadata хранятся числа, строки, списки и ссылки на большие артефакты. CSV подходит для итоговой плоской таблицы, JSON — для вложенных components и assumptions.

Большинство estimator-ов работают после обучения, но prequential estimator управляет серией обучений. Поэтому `BenchmarkRunner` предоставляет контролируемую функцию обучения; общий API при этом сохраняет единый формат результата, а не обязательно совершенно одинаковую вычислительную процедуру.

### Связка

«Теперь разберём четыре алгоритмические ветки. Первая — дешёвые информационные критерии и явная двухчастная MDL-кодировка.»

### Вероятные вопросы

**Почему нужен registry, если estimator можно создать напрямую?**  
Для библиотечного API прямое создание остаётся. Registry позволяет из конфигурации воспроизводимо собрать набор методов без большого `if/else`.

**Будет ли зависимость от конкретной posterior-библиотеки?**  
В ядре — нет. Нужен минимальный adapter protocol: samples или распределение `q`, pointwise log-likelihood и при необходимости KL. Конкретные интеграции будут опциональными.

**Почему Python 3.11+ и PyTorch?**  
Это практический выбор для типизации, packaging и прямой работы с небольшими сетями. Теория от PyTorch не зависит, но первая реализация ограничена им.

**Как учесть, что некоторые методы требуют переобучения, а другие нет?**  
В result хранятся runtime, peak memory и число fits. Benchmark сравнивает не только score, но и дополнительную стоимость относительно базового обучения.

---

## Слайд 6. Algorithms I: information criteria and two-part MDL

**Рекомендуемое время:** 1 минута 20 секунд.

### Что сказать

«Первая ветка объединяет дешёвые критерии и explicit two-part code. Пусть
$\widehat L$ — максимизированный total likelihood, $k$ — число оценённых свободных
параметров, а $n$ — число независимых наблюдений. У AIC штраф $2k$, у BIC —
$k\log n$, у HQIC — $2k\log\log n$. Во всех случаях меньшее значение
предпочтительно: критерий балансирует fit через $-2\log\widehat L$ и штраф сложности.
Замена $k$ на effective dimension возможна только как отдельно названная и
документированная модификация, а не как стандартный AIC/BIC/HQIC.

WAIC использует posterior predictive fit — lppd — и effective number of parameters $p_{\mathrm{WAIC}}$. WBIC оценивает free energy через ожидание negative log-likelihood по tempered posterior, обычно с inverse temperature $1/\log n$. Для WAIC и WBIC одной MAP-модели недостаточно.

В two-part MDL сначала кодируются квантизованные параметры $\theta_\delta$, затем данные при этих параметрах: $L(\theta_\delta)+L(D\mid\theta_\delta)$. Precision $\delta$ является частью схемы кодирования. Эти методы мы сначала проверим на малых GLM, где есть аналитические или эталонные значения.»

### AIC, BIC и HQIC

Пусть $\ell(\theta)=\log p(D\mid\theta)$, а $\hat\theta$ максимизирует $\ell$.

\[
\mathrm{AIC}=-2\ell(\hat\theta)+2k.
\]

**AIC** выводится как асимптотическая поправка к оптимистичности in-sample log-likelihood и ориентирован на predictive accuracy, то есть ожидаемое KL-расхождение от истинного распределения к fitted model. Это не вероятность модели и не consistent model selector в общем случае.

\[
\mathrm{BIC}=-2\ell(\hat\theta)+k\log n.
\]

**BIC** получается из Laplace approximation к marginal likelihood при фиксированной конечной размерности, regular identifiable model, гладком likelihood, независимых наблюдениях и prior, положительном около optimum. С точностью до $O(1)$:

\[
-2\log p(D\mid M)\approx \mathrm{BIC}.
\]

Для глубоких сетей regularity и identifiability нарушаются из-за перестановочных симметрий, масштабных симметрий и плоских направлений. Поэтому BIC — baseline с warning, а не гарантированно корректная evidence approximation.

\[
\mathrm{HQIC}=-2\ell(\hat\theta)+2k\log\log n.
\]

**HQIC** исторически разработан для consistent order selection во временных рядах. Его penalty растёт быстрее AIC и медленнее BIC. В нашем benchmark это сравнительный baseline; теоретическую гарантию нельзя автоматически переносить на нейросети.

**Что такое $k$.** Прозрачный baseline — число trainable scalar parameters. «Effective $k$» разрешён только с явным определением. Нельзя молча заменить $k$ рангом Hessian или числом ненулевых весов и назвать результат стандартным AIC/BIC.

### WAIC

Для posterior draws $\theta^{(s)}\sim p(\theta\mid D)$:

\[
\mathrm{lppd}
=\sum_{i=1}^{n}\log\left(
\mathbb E_{p(\theta\mid D)}[p(y_i\mid x_i,\theta)]
\right),
\]

\[
p_{\mathrm{WAIC}}
=\sum_{i=1}^{n}\operatorname{Var}_{p(\theta\mid D)}
\left[\log p(y_i\mid x_i,\theta)\right],
\]

\[
\mathrm{WAIC}=-2(\mathrm{lppd}-p_{\mathrm{WAIC}}).
\]

`lppd` — **log pointwise predictive density**. Усреднение вероятностей по posterior выполняется **до** логарифма. $p_{\mathrm{WAIC}}$ оценивает effective number of parameters через вариативность pointwise log-likelihood. WAIC асимптотически связан с Bayesian leave-one-out predictive performance и применим также к singular models, но практическая точность зависит от posterior approximation и числа samples.

### WBIC

Определим tempered posterior:

\[
p_\beta(\theta\mid D)
\propto p(D\mid\theta)^\beta p(\theta),
\qquad \beta=1/\log n.
\]

Тогда

\[
\mathrm{WBIC}
=\mathbb E_{p_\beta(\theta\mid D)}[-\log p(D\mid\theta)]
\]

асимптотически оценивает Bayes free energy $-\log p(D)$. WBIC требует inference именно при температуре $\beta$, а не просто samples из обычного posterior $\beta=1$. При $\beta<1$ likelihood уплощается.

### Two-part MDL

\[
L_{2p}(D,\theta_\delta)
=L(\theta_\delta)+L(D\mid\theta_\delta).
\]

Первая часть сообщает декодеру архитектуру и квантизованные параметры; вторая кодирует labels при известных inputs. При идеальном Shannon code $L(D\mid\theta_\delta)=-\log_2p(D\mid\theta_\delta)$ bits. Реальный **prefix code** — код, в котором ни одно кодовое слово не является префиксом другого; это обеспечивает однозначное декодирование, а длины удовлетворяют неравенству Крафта.

Без фиксированных precision $\delta$, диапазона, порядка параметров, кода архитектуры и encoder/decoder $L(\theta_\delta)$ не определена. Слишком высокая precision делает parameter code дорогим; слишком низкая ухудшает data code. Если $\delta$ выбирается по данным из сетки, стоимость сообщения этого выбора тоже должна учитываться, иначе получается оптимистичная оценка.

### Связка

«Первая группа использует maximum likelihood, простые penalties или explicit quantization. Вторая учитывает распределение по параметрам и локальную геометрию posterior.»

### Вероятные вопросы

**Чем AIC принципиально отличается от BIC?**  
AIC нацелен на out-of-sample prediction и имеет penalty $2k$. BIC приближает $-2\log$ marginal likelihood и имеет penalty $k\log n$. Их цели различны: prediction против evidence/model identification в regular setting.

**Корректны ли AIC и BIC для deep networks?**  
Стандартные выводы предполагают regular identifiable models и обычно фиксированное $k$ при $n\to\infty$. Нейросети singular и переопределены, поэтому строгая применимость нарушена. Мы используем их как дешёвые baselines и явно отмечаем ограничение.

**Почему WAIC одной MAP-сети недостаточно?**  
Нужны posterior expectation predictive density и posterior variance pointwise log-likelihood. Одна точка не позволяет оценить ни усреднение, ни variance.

**Чем WBIC отличается от variational free energy?**  
WBIC — ожидание NLL при tempered posterior с $\beta=1/\log n$. Variational free energy — upper bound на $-\log p(D)$ относительно выбранного $q$, содержащий явный KL к prior.

**Почему two-part MDL не равен размеру `.pt`?**  
Файл содержит заголовки, serialization overhead и иногда optimizer state. Теоретическая длина требует заранее определённого decodable code; файловый формат не является такой моделью кода автоматически.

---

## Слайд 7. Algorithms II: Bayesian coding and marginal evidence

**Рекомендуемое время:** 1 минута 25 секунд.

### Что сказать

«Во второй ветке есть два приближения к байесовской цене модели. Variational coding использует отрицательный ELBO: ожидаемая NLL под $q_\phi(\theta)$ плюс $\mathrm{KL}(q_\phi\|p)$. Первый член описывает данные, второй — сколько информации нужно, чтобы перейти от prior к приближённому posterior.

Laplace approximation локально заменяет posterior около MAP нормальным распределением. Negative log evidence складывается из negative log joint в MAP и curvature correction: половины log determinant Hessian минус $d/2\log(2\pi)$. Полная Hessian слишком велика, поэтому KFAC приближает curvature послойными Kronecker factors.

Оба результата чувствительны к prior и качеству аппроксимации; Laplace дополнительно чувствителен к выбору curvature matrix и damping. Мы будем сохранять все эти настройки и компоненты score.»

### Variational coding, ELBO и KL

Bayes evidence, или marginal likelihood:

\[
p(D)=\int p(D\mid\theta)p(\theta)\,d\theta.
\]

Он усредняет likelihood по prior и автоматически штрафует модели, которые хорошо fit-ят данные только в очень малой области parameter space. Точный интеграл для сети обычно недоступен.

Вводим tractable approximation $q_\phi(\theta)$. Тогда:

\[
\mathrm{ELBO}(q)
=\mathbb E_q[\log p(D\mid\theta)]
-\mathrm{KL}(q(\theta)\|p(\theta)),
\]

\[
-\mathrm{ELBO}(q)
=\mathbb E_q[-\log p(D\mid\theta)]
+\mathrm{KL}(q(\theta)\|p(\theta)).
\]

Точная identity:

\[
\log p(D)=\mathrm{ELBO}(q)
+\mathrm{KL}\bigl(q(\theta)\|p(\theta\mid D)\bigr).
\]

Так как KL неотрицателен, ELBO — lower bound на log evidence, а negative ELBO — upper bound на negative log evidence. Gap равен KL от $q$ к истинному posterior.

Определение KL:

\[
\mathrm{KL}(q\|p)
=\mathbb E_q\left[\log\frac{q(\theta)}{p(\theta)}\right]\ge0.
\]

Первый член negative ELBO называется expected data cost или expected NLL. Второй — complexity cost относительно **конкретного prior**. Изменение масштаба prior меняет KL и итоговый score, поэтому prior нельзя подбирать на test set.

**Bits-back caveat.** Negative ELBO имеет кодовую интерпретацию в variational/bits-back coding, но это не означает, что простое сложение двух floating-point чисел уже является реализованным compressor-ом. Для достижения такой длины нужны код для latent parameters, общий prior, начальные bits или amortization и конкретный bits-back protocol. Поэтому в первой версии корректнее назвать результат variational description-length surrogate и отдельно указать, реализован ли фактический encoder/decoder.

### KFAC–Laplace evidence

Пусть $\hat\theta$ — MAP:

\[
\hat\theta=\arg\max_\theta
\left[\log p(D\mid\theta)+\log p(\theta)\right].
\]

Определим negative log joint

\[
h(\theta)=-\log p(D,\theta)
=-\log p(D\mid\theta)-\log p(\theta),
\]

и curvature в MAP

\[
H=\nabla^2_\theta h(\theta)\big|_{\theta=\hat\theta}.
\]

Квадратическое разложение даёт posterior approximation

\[
p(\theta\mid D)\approx
\mathcal N(\hat\theta,H^{-1})
\]

и

\[
-\log p(D)
\approx -\log p(D,\hat\theta)
+\frac12\log\det H
-\frac d2\log(2\pi).
\]

Здесь $d$ — размерность параметров. $\log\det H$ суммирует log-curvatures по направлениям. Интуитивно узкий optimum имеет большие eigenvalues и сильный Occam penalty; широкий optimum — меньший. Но интерпретация зависит от параметризации и prior.

Для слоя KFAC приближает Fisher или generalized Gauss–Newton block как

\[
F_l\approx G_l\otimes A_l,
\]

где $A_l$ — covariance входных activations, $G_l$ — covariance градиентов pre-activations, $\otimes$ — Kronecker product. Это позволяет дешевле хранить curvature и считать log determinant через eigenvalues factors.

**Ключевая оговорка:** KFAC обычно приближает Fisher/GGN, а не точную Hessian negative log joint. Для likelihood-моделей GGN даёт положительно полуопределённую матрицу, удобную для Gaussian approximation, но итоговая formula является дополнительной аппроксимацией. Prior precision должна быть добавлена к curvature согласованно.

**Damping** — добавление положительного сдвига, например $H_\lambda=H+\lambda I$, чтобы matrix была положительно определённой и log determinant стабилен. Damping меняет evidence, поэтому это не только numerical trick: нужна sensitivity analysis по $\lambda$.

### Связка

«Bayesian branch кодирует цену posterior через prior или локальную curvature. Следующая ветка вообще избегает передачи весов и кодирует labels последовательными предсказаниями learner-а.»

### Вероятные вопросы

**Почему negative ELBO можно считать сложностью, если в нём есть data-fit term?**  
Это полная description-length/evidence surrogate, а не чистая parameter complexity. Мы отдельно возвращаем expected NLL и KL. Для анализа «штрафа сложности» можно смотреть KL отдельно, но модельный критерий — их сумма.

**Какой prior вы выберете?**  
Начальный baseline — заранее фиксированный factorized zero-mean Gaussian с явно указанной variance или precision. Затем проводится sensitivity analysis. Prior нельзя выбирать по test metric; если hyperparameter обучается по train/evidence, процедура должна быть зафиксирована.

**Что такое MAP и чем он отличается от MLE?**  
MLE максимизирует $p(D\mid\theta)$. MAP максимизирует $p(D\mid\theta)p(\theta)$, то есть учитывает prior. При Gaussian prior MAP связан с L2 regularization.

**Что делать с нулевыми или отрицательными eigenvalues Hessian?**  
Точная Gaussian Laplace approximation требует положительно определённой curvature. На практике используют PSD GGN/Fisher и prior precision, плюс damping. Если положительная определённость не достигнута, estimator должен вернуть warning или отказ, а не скрывать `NaN` через произвольный модуль determinant.

**Почему KFAC log-det дешевле полного?**  
Полная matrix имеет размер $d\times d$, хранение $O(d^2)$, decomposition $O(d^3)$. Для Kronecker block eigenvalues — попарные произведения eigenvalues малых factors, поэтому log-det можно вычислить через их суммы без materialization полной matrix.

---

## Слайд 8. Algorithms III: prequential coding

**Рекомендуемое время:** 1 минута 10 секунд.

### Что сказать

«Prequential coding кодирует labels в том порядке, в котором learner получает данные. Dataset разбивается на последовательные блоки. Первый блок кодируется равномерным кодом — до обучения модель ещё ничего не знает. Затем модель обучается только на уже декодированных данных и предсказывает следующий блок; длина кода равна сумме negative log-probabilities правильных labels.

Такой score измеряет не только финальный fit, но и скорость, с которой конкретная процедура обучения извлекает структуру из данных. Мы реализуем blockwise baseline и, если позволит время, online/replay variant. Стоимость метода включает повторные fits. Для воспроизводимости фиксируются порядок данных, split, block schedule, seed, training budget и calibration.»

### Формула и обозначения

Пусть $0=n_0<n_1<\dots<n_m=n$, а $K$ — число классов:

\[
L_{\mathrm{preq}}
=n_1\log_2K
-\sum_{j=2}^{m}\sum_{i=n_{j-1}+1}^{n_j}
\log_2p_{\hat\theta(D_{<j})}(y_i\mid x_i).
\]

- $D_{<j}$ — только объекты из предыдущих блоков;
- $\hat\theta(D_{<j})$ — результат обучения по этим данным;
- $n_1\log_2K$ — uniform code первого блока;
- следующий член — predictive code остальных labels.

Если модель присваивает истинной метке вероятность $p$, оптимальная идеализированная длина сообщения равна $-\log_2p$. Хорошо калиброванное уверенное правильное предсказание короткое; уверенная ошибка очень дорогая.

### Полный encoder/decoder protocol

Чтобы это была реальная кодовая схема, обе стороны заранее согласуют:

1. inputs $x_i$, порядок объектов и границы блоков;
2. архитектуру, initialization rule, optimizer, число epochs или stopping rule, preprocessing и seed;
3. finite-precision probability coder, например arithmetic/range coding.

Далее encoder:

1. отправляет labels первого блока uniform code;
2. encoder и decoder независимо обучают одинаковую модель на уже известных парах;
3. encoder кодирует labels следующего блока с предсказанными probabilities;
4. decoder восстанавливает labels, добавляет их к train data и повторяет цикл.

Это объясняет запрет leakage: модель, calibration и stopping для блока $j$ не могут использовать labels этого или будущих блоков. Test set проекта вообще не участвует в prequential code и остаётся только для внешней оценки связи score с generalization.

**Blockwise** переобучает с нуля на каждом растущем prefix. **Online/replay** продолжает обучение и переигрывает прошлые данные; это дешевле, но score сильнее зависит от stateful training protocol. Эти варианты нельзя смешивать под одним именем.

**Calibration.** Temperature scaling допустим только по ранее доступным данным. Калибровка на кодируемом или test block создаёт leakage. Мы сравним raw и causally calibrated probabilities, сохраняя протокол.

**GPU nondeterminism caveat.** Даже одинаковый seed не всегда гарантирует bit-identical обучение на GPU из-за nondeterministic kernels и порядка floating-point operations. Для буквального decoder protocol нужны deterministic algorithms/CPU либо передача дополнительных сведений. Если гарантируется только статистическая воспроизводимость, результат нужно честно назвать estimated prequential length, а не утверждать реальный round-trip code.

### Связка

«Prequential MDL кодирует данные через способность learner-а быстро учиться. Четвёртая ветка, наоборот, строит явное сжатое представление уже обученной сети.»

### Вероятные вопросы

**Почему первый блок кодируется равномерно?**  
До получения labels learner не имеет обучающих данных. Uniform code — простой честный default стоимостью $\log_2K$ bits на label. Можно использовать согласованный prior по классам, но его источник и стоимость должны быть заданы заранее.

**Зависит ли prequential score от порядка данных?**  
Да. Поэтому порядок фиксируется; для оценки устойчивости можно повторить несколько случайных permutations и отчитываться о среднем и разбросе.

**Это сложность модели или алгоритма обучения?**  
Скорее сложность пары «model class + learner + training protocol» на конкретных данных. В этом и смысл: быстро обучаемая структура получает короткий код.

**Почему нельзя кодировать test set как последний блок, а затем использовать тот же test NLL?**  
Тогда целевая метрика и complexity score используют одни и те же test labels, возникает leakage и механическая корреляция. Prequential code строится только внутри training partition; held-out test остаётся независимым.

**Что будет, если модель выдаст нулевую вероятность истинного класса?**  
Кодовая длина станет бесконечной. На практике probabilities вычисляются устойчиво через log-softmax и, для finite coder, ограничиваются заранее объявленным минимальным уровнем; это ограничение меняет code и должно быть задокументировано.

---

## Слайд 9. Algorithms IV: compression-based complexity

**Рекомендуемое время:** 55–65 секунд.

### Что сказать

«В compression branch обученная сеть заменяется дискретным сжатым представлением $C$ с контролируемым ухудшением качества. Мы рассматриваем pruning, low-rank approximation и quantization. На слайде разделены два объекта. Первый — полный two-part data code $L(C)+L(D\mid C)$: сначала передаётся модель, затем данные при известной модели. Второй — только model description length $\ell(C)$ в bits, которая входит в Occam bound.

Для bounded loss показана конкретная схематическая форма: истинный риск сжатой модели не превосходит её empirical risk плюс корень из суммы $\ell(C)\ln 2$ и confidence term, делённой на $2n$. Здесь принципиально, что bound относится к декодированной **сжатой модели**, а не автоматически к исходной сети. Encoder, decoder, precision и допустимое искажение фиксируются заранее; размер файла `.pt` не заменяет теоретическую длину кода.»

### Формула и смысл

\[
\underbrace{L(C)+L(D\mid C)}_{\text{two-part data code}}
\qquad\text{и}\qquad
\underbrace{\ell(C)}_{\text{model description in bits}}.
\]

Здесь $C$ должно быть полным декодируемым представлением: например, sparse mask, индексы и quantized nonzero values; для low-rank — ranks и factors; для quantization — codebook, assignments и все side information. В two-part code величина $L(D\mid C)$ кодирует данные при уже декодированной модели. Она не входит автоматически в model-description term теоремы об обобщении.

Для prefix-coded hypothesis и loss в диапазоне $[0,1]$ на слайде используется Occam/union-bound template:

\[
R(C)\leq \widehat R(C)
+\sqrt{\frac{\ell(C)\ln 2+\ln(1/\delta)}{2n}}.
\]

- $R(C)$ — population risk декодированной сжатой модели;
- $\widehat R(C)$ — empirical risk той же модели;
- $\ell(C)$ — длина её префиксного описания именно в **bits**;
- множитель $\ln 2$ переводит эту длину в nats внутри exponential bound;
- $n$ — число независимых training examples;
- $\delta$ — confidence level: с вероятностью не менее $1-\delta$ выполняется bound.

Коэффициент $1/2$ и сама форма зависят от выбранной concentration inequality и assumptions, поэтому перед реализацией нужно зафиксировать конкретную theorem. Для unbounded NLL эта формула напрямую неприменима: нужны clipping, tail assumptions или другая граница.

**Почему bound относится к compressed model.** Теорема контролирует гипотезу, которую можно описать кодом длины $\ell(C)$, то есть decoder output. Чтобы перенести вывод на исходную сеть, нужно дополнительно ограничить расхождение её predictions с compressed network; просто близость весов недостаточна.

**Compression ratio** нужно определять явно, например original parameter bits / encoded bits. Несколько levels дают rate–distortion curve: сколько bits требуется при данном ухудшении empirical risk.

### Связка

«Мы получили четыре семейства с разной семантикой: predictive criteria, evidence surrogates, literal/sequential codes и compression bounds. Поэтому benchmark должен сравнивать их аккуратно, сохраняя native units и assumptions.»

### Вероятные вопросы

**Почему небольшой compressed file означает хороший bound?**  
Не сам по себе. Bound содержит и empirical risk compressed model. Чрезмерное сжатие уменьшит $\ell(C)$, но может увеличить $\widehat R(C)$; нужен баланс rate–distortion.

**Можно ли использовать ZIP-размер checkpoint?**  
Как инженерную эвристику — отдельно. Для теоретического Occam bound нужен заранее заданный prefix-free или uniquely decodable code и полный side information. Общий compressor и формат файла могут включать неизвестный overhead.

**Что значит pruning в кодовой стоимости?**  
Нужно закодировать mask или индексы ненулевых weights, сами quantized values, их precision, размеры layers и codebook. Считать только число ненулевых weights недостаточно.

**Гарантирует ли bound качество исходной несжатой сети?**  
Нет, непосредственно он относится к compressed/decompressed model. Для исходной сети нужна отдельная связь через контролируемое изменение predictions или risk.

**Можно ли считать показанную формулу уже сертифицированной границей для NLL?**  
Нет. На слайде указан Occam template для bounded loss и prefix code. В коде будет выбрана конкретная theorem с её constants и assumptions; для NLL потребуется отдельное обоснование, потому что NLL не ограничена сверху.

---

## Слайд 10. Benchmark protocol and hypotheses

**Рекомендуемое время:** 1 минута.

### Что сказать

«Benchmark фиксирует данные, splits, preprocessing, optimizer, stopping rule и compute budget, а варьирует ширину, глубину, регуляризацию, размер training set и seed по заранее заданной сетке. Для каждой обученной модели мы сохраняем parameter count, train и validation NLL, все complexity scores в native units и, где допустимо, в nats или bits per sample. Отдельно измеряются runtime, число fits и peak memory.

Основная метрика — Spearman rank correlation complexity score с held-out test NLL и NLL generalization gap. Ранговая корреляция выбрана потому, что шкалы методов различаются и связь может быть монотонной, но нелинейной. Мы также проверяем стабильность ranking по seeds и sample sizes. Test set используется только для финального вычисления целевых метрик, но не для настройки моделей, prior, compression, calibration или estimator hyperparameters.»

### Экспериментальная единица и матрица запусков

Одна строка benchmark должна соответствовать однозначно определённому объекту:

\[
(\text{dataset version},\text{split},n,\text{architecture},
\text{regularization},\text{training seed},\text{estimator config}).
\]

Чтобы корреляции не объяснялись случайным дисбалансом, для каждого сочетания
architecture/regularization желательно использовать одинаковый набор seeds и splits.
Post-hoc estimator-ы запускаются на общих checkpoints. Методы с auxiliary inference,
compression или repeated training получают matched dataset/model/training configuration.
Именно общий experimental unit и согласованные факторы образуют paired design; один и
тот же checkpoint не является обязательным входом каждого метода.

**Данные:** synthetic classification, MNIST, Fashion-MNIST. Train/validation/test indices создаются один раз и сохраняются. Если варьируется $n$, малые train sets желательно делать вложенными prefixes одного зафиксированного порядка, чтобы уменьшить лишнюю вариативность.

**Модели:** MLP/CNN; варьируются width, depth и регуляризация, например weight decay/dropout. Менять всё сразу без factorial design опасно: нельзя будет понять, какой фактор связан со score.

**Обучение:** фиксируются optimizer, learning-rate schedule, maximum epochs, early stopping criterion и compute budget. Early stopping использует только validation, не test.

**MLE-like baseline:** стандартные AIC/BIC/HQIC используют максимизированный
**total likelihood**. Checkpoint после weight decay, dropout, data augmentation или
early stopping не является буквальным MLE этой likelihood. Поэтому для проверки
стандартной формулы нужен отдельный контролируемый MLE-like run; применение критерия
к regularized checkpoint маркируется как heuristic и сопровождается warning. Если
PyTorch loss усреднён по batch/датасету, total log likelihood восстанавливается через
сумму по всем observations до добавления penalty.

**Baselines:** parameter count, train NLL, validation NLL. Без них нельзя понять, даёт ли сложный estimator больше информации, чем дешёвая величина.

### Test NLL и generalization gap

В качестве основных целевых величин используем средние NLL:

\[
\overline{\mathrm{NLL}}_{S}
=-\frac1{|S|}\sum_{i\in S}\ln p_{\hat\theta}(y_i\mid x_i),
\]

\[
g_{\mathrm{NLL}}
=\overline{\mathrm{NLL}}_{\mathrm{test}}
-\overline{\mathrm{NLL}}_{\mathrm{train}}.
\]

Используется одна и та же checkpoint и один и тот же evaluation mode. Для dropout/batch normalization вызывается `model.eval()`. Среднее, а не сумма, позволяет сравнивать splits разных размеров. Дополнительно можно считать accuracy gap, но он анализируется отдельно.

### Spearman correlation

Для $N$ моделей complexity values $c_i$ и target values $t_i$ преобразуются в ranks $R(c_i)$ и $R(t_i)$. Затем:

\[
\rho_s=\operatorname{Corr}_{\mathrm{Pearson}}
\left(R(c),R(t)\right).
\]

Если ties отсутствуют, эквивалентная формула:

\[
\rho_s=1-\frac{6\sum_i d_i^2}{N(N^2-1)},
\quad d_i=R(c_i)-R(t_i).
\]

При ties используем стандартное усреднение ranks и Pearson correlation ranks, а не упрощённую формулу. $\rho_s=1$ означает одинаковый монотонный порядок, $-1$ — обратный, около нуля — отсутствие монотонной связи. Для criteria, где меньше лучше, положительная связь с test NLL обычно ожидаема: высокий/плохой criterion соответствует высокой/плохой NLL. Если мы выделяем только complexity penalty, знак интерпретируется отдельно.

**Почему Spearman, а не только Pearson.** Methods имеют разные шкалы и могут быть связаны с quality нелинейно. Spearman устойчив к монотонным transformations, включая перевод nats в bits. Но он теряет информацию о величине различий, поэтому scatter plots и Pearson как secondary analysis полезны.

**Uncertainty.** Не следует сообщать одно $\rho$ без разброса. Минимум — correlations по независимым seeds/splits и bootstrap confidence interval, причём bootstrap лучше делать на уровне model configurations или групп, сохраняя зависимость повторов.

### Защита от leakage

До запуска фиксируются analysis plan и grids. Test labels нельзя использовать для:

- выбора architecture, checkpoint или early stopping;
- выбора prior variance, damping, quantization precision или block schedule;
- probability calibration;
- отбора «лучшей» версии estimator-а;
- решения, какие неудачные seeds исключить.

Hyperparameters выбираются на theory grounds или train/validation. Затем test метрики вычисляются один раз для frozen configurations. Если test многократно используется в ходе разработки, он фактически становится validation set; тогда нужен новый untouched holdout.

### Связка

«Первый сквозной запуск будет намеренно меньше полного benchmark: он проверит API, воспроизводимость и весь data path до того, как мы реализуем дорогие варианты.»

### Вероятные вопросы

**Почему Spearman, если complexity score должен численно предсказывать NLL?**  
У методов нет общей калиброванной шкалы: некоторые возвращают criterion, некоторые code length, некоторые bound. Первый честный вопрос — правильно ли они ранжируют модели. Численную calibration можно анализировать вторично внутри сопоставимых семейств.

**Что является одной точкой при расчёте корреляции?**  
Одна frozen experimental configuration: конкретные data split, architecture,
regularization, training seed и estimator protocol с соответствующей held-out target.
Для post-hoc метода в неё входит trained checkpoint; для prequential или posterior
workflow — matched auxiliary inference либо repeated fits. Повторные seeds не следует
бездумно считать полностью независимыми; мы покажем результаты по группам и uncertainty.

**Почему generalization gap может быть отрицательным?**  
Из-за сильной data augmentation/dropout на train, различий режима вычисления или статистического шума test NLL иногда может быть ниже train NLL. Поэтому evaluation protocol должен унифицировать loss и режим; сам отрицательный gap не запрещён.

**Как избежать того, что размер модели одновременно повышает и complexity, и train quality?**  
Использовать factorial/stratified analysis: correlations внутри фиксированных dataset sizes и регуляризации, partial/regression analyses как secondary checks, и обязательно сравнить с parameter-count baseline.

**Можно ли объединить все datasets в одну корреляцию?**  
Не как основную метрику: различия datasets и NLL scales могут создать ложную корреляцию. Сначала считаются результаты внутри dataset/sample-size strata, затем агрегируются с явной процедурой.

---

## Слайд 11. Proof-of-concept demonstration

**Рекомендуемое время:** 40–45 секунд.

### Что сказать

«Для Tech Meeting 2 мы планируем небольшой end-to-end proof of concept: обучить три–пять MLP разной ширины на одном фиксированном MNIST split и через общий API вычислить parameter count, BIC, two-part MDL и blockwise prequential code. Мы сохраним config, seed, checkpoints, компоненты score, время и память, затем построим таблицу и scatter plot против test NLL и generalization gap.

Цель PoC — проверить контракт и полный путь данных, а не заранее получить сильные научные выводы. Финальное demo принимает dataset split, checkpoint и список estimator-ов и возвращает таблицу, ranking, decomposition, runtime и warnings. Один CLI или notebook запускается в чистом окружении.»

### Детали PoC

**Почему 3–5 моделей недостаточно для вывода.** Это smoke test: на столь малом $N$ correlation крайне нестабильна. По результатам PoC можно утверждать, что pipeline работает, но не что один estimator лучше другого.

**Почему именно BIC, two-part MDL и prequential.** Они покрывают три разные computational patterns: дешёвый post-hoc formula, explicit parameter/data code и repeated training. Если общий API выдерживает эти случаи, расширение другими методами проще.

**Минимальные acceptance checks:**

- повторный запуск с тем же config даёт то же значение в пределах объявленной tolerance;
- все units и components присутствуют;
- BIC совпадает с ручным расчётом на toy example;
- two-part decoder восстанавливает квантизованную модель;
- prequential blocks не используют future labels;
- таблица и plot строятся из сохранённого result, а не из скрытого notebook state.

### Связка

«Эти deliverables распределены между четырьмя участниками так, чтобы у каждой алгоритмической ветки был один ответственный, а интеграция проходила cross-review.»

### Вероятные вопросы

**Почему в PoC нет WAIC и KFAC-Laplace?**  
Они требуют отдельной posterior/curvature infrastructure. PoC сначала проверяет сквозной контракт дешёвыми и структурно различными методами; затем эти adapters добавляются без изменения формата результата.

**Что покажет demo, если методы несопоставимы по абсолютной шкале?**  
Native score и units, допустимую normalization, decomposition, ranks внутри протокола и warnings. Demo не будет скрывать несопоставимость за одним безымянным столбцом.

**Почему Streamlit/Notebook, а не web service?**  
Для учебного проекта важнее воспроизводимый локальный сценарий. Web service добавляет deployment work, не улучшающий проверку алгоритмов.

---

## Слайд 12. Team responsibilities and algorithm ownership

**Рекомендуемое время:** 40–45 секунд.

### Что сказать

«У каждой ветки один основной владелец. Алябушев отвечает за AIC, BIC, HQIC, WAIC, WBIC и two-part MDL, а также за benchmark и PoC. Дементьев — за variational coding, KFAC–Laplace, упаковку библиотеки и тестовую инфраструктуру. Курдюков — за blockwise и online/replay prequential coding, документацию и demo. Чирков — за compression-based score, выбранный generalization bound, blog post и technical report.

Планирование остаётся общей обязанностью. Для cross-review используется цикл, показанный на слайде. Готовая алгоритмическая ветка — это не только формула: она включает assumptions, реализацию общего интерфейса, минимальный пример, unit tests и документацию.»

### Что означает ownership

Владелец метода должен предоставить:

1. математическую спецификацию с units и assumptions;
2. estimator, возвращающий общий `ComplexityResult`;
3. toy oracle или ручной пример;
4. тесты ошибок и граничных случаев;
5. config для benchmark;
6. краткую документацию и known limitations.

**Cross-review** нужен не только для style. Reviewer проверяет формулу, отсутствие leakage, единицы, возможность воспроизведения и соответствие API. Автор остаётся ответственным за исправления.

### Связка

«Чтобы параллельная работа сошлась в единый пакет, для каждого этапа заранее задан проверяемый интеграционный результат.»

### Вероятные вопросы

**Почему у Алябушева больше названий алгоритмов?**  
AIC/BIC/HQIC — небольшие формульные baselines с общей инфраструктурой likelihood и parameter counting. Более короткий список у других участников включает существенно более тяжёлые posterior, curvature, repeated-training или compression pipelines.

**Кто отвечает за интеграцию веток?**  
Дементьев — за package/API и tests, Алябушев — за benchmark. Но merge требует cross-review, а каждый владелец обязан реализовать общий контракт.

**Что произойдёт, если дорогой метод не будет готов?**  
Минимальный baseline каждой ветки должен быть определён заранее. Например, blockwise prequential раньше online/replay, diagonal или small-layer Laplace раньше полного KFAC grid. Missing method не должен блокировать работающий package.

---

## Слайд 13. Milestones and acceptance criteria

**Рекомендуемое время:** 40–45 секунд.

### Что сказать

«На первой встрече фиксируются scope, формулы, units, architecture, API, роли и PoC plan. Ко второй нужен end-to-end PoC и черновики документации, blog post и technical report. Checkpoint требует устанавливаемого пакета, общего code style, workflows и cross-review. К третьей встрече должны работать все ветки, общий benchmark, coverage выше 90 процентов, demo и итоговый report.

Definition of done проверяет установку из чистого окружения, единый контракт всех estimator-ов, полную фиксацию данных и единиц в benchmark, tests формул и edge cases и воспроизводимые примеры в документации.»

### Как понимать критерии

**Coverage > 90%** — полезный сигнал, но не доказательство корректности. Важнее содержательные tests: known analytical values, invariance к batch partition, unit conversion, missing-artifact errors, deterministic replay и end-to-end smoke test.

**Clean environment** означает новую virtual environment без неявных локальных зависимостей. Команда установки, версия Python и команда запуска demo должны быть в README.

**Workflow** — автоматический lint/type/test/build pipeline при изменении кода. Он не должен зависеть от GPU для базовых tests.

**Definition of done на estimator:** формула и citation; native scale; inputs; failure modes; implementation; unit tests; reproducible example; benchmark config.

### Связка

«Основные угрозы срокам и корректности уже видны из предпосылок методов; следующий слайд показывает, как мы их контролируем.»

### Вероятные вопросы

**Почему технический отчёт начинается до завершения экспериментов?**  
Потому что постановка, формулы, protocol и limitations не зависят от финальных чисел. Ранний draft выявляет неясные определения до реализации.

**Что означает end-to-end PoC?**  
От сохранённого config и чистого окружения через обучение/загрузку checkpoint и вызов estimator-ов до сериализованной таблицы и графика — без ручного переноса чисел.

**Достаточен ли coverage 90%?**  
Нет. Это количественный порог, дополняющий, а не заменяющий oracle tests и научную валидацию.

---

## Слайд 14. Main risks and controls

**Рекомендуемое время:** 45–50 секунд.

### Что сказать

«Основные риски связаны не с синтаксисом реализации, а с неправильной интерпретацией. AIC/BIC могут использовать некорректную размерность для singular neural network; поэтому parameter count остаётся явно помеченным baseline. WAIC/WBIC требуют posterior inference и не вычисляются из одного MAP checkpoint. KFAC/Laplace чувствителен к prior, curvature и damping, поэтому мы сохраняем decomposition и делаем sensitivity analysis.

Prequential coding требует многократного обучения, поэтому начинаем с малых моделей и фиксированного block schedule. Compression score зависит от конкретного кода, поэтому проверяем round-trip. Разные native scales не смешиваются без обоснованного перевода. Из-за ограниченных ресурсов сначала реализуется минимальный MLP baseline, затем CNN и полный grid.»

### Карта «риск → наблюдаемый симптом → контроль»

- **Некорректный $k$:** BIC монотонно огромен и просто повторяет parameter count. Контроль: показывать fit/penalty отдельно, маркировать regularity caveat, сравнивать с простыми GLM.
- **Плохие posterior samples:** WAIC variance шумная, effective parameter count нестабилен. Контроль: diagnostics, число samples, repeated estimates; не возвращать score при отсутствии обязательных данных.
- **Неустойчивая curvature:** `logdet` даёт `NaN` или резко меняется от damping. Контроль: PSD backend, prior precision, eigenvalue diagnostics, sweep по damping.
- **Prequential leakage:** неожиданно короткий code первого predictive block. Контроль: audit indices, immutable split, test с deliberately permuted future labels.
- **Неполный compression code:** длина не учитывает mask/codebook. Контроль: настоящий encoder/decoder, подсчёт всех bitstreams и round-trip test.
- **Ложное ранжирование из-за units:** сумма и mean, bits и nats смешаны. Контроль: typed units, native value и explicit conversion.

### Связка

«При этих ограничениях итог проекта формулируется не как одна магическая метрика, а как прозрачная платформа сравнения с явными предпосылками.»

### Вероятные вопросы

**Какой риск самый критичный?**  
Скрытая несопоставимость: одинаковая колонка `complexity` может смешать разные units и meanings. Поэтому structured result, assumptions и component decomposition — часть корректности, а не косметика.

**Что вы сократите первым при нехватке времени?**  
Расширенные grids, CNN и online/replay variants. Не сокращаются общий API, единицы, базовые oracle tests и защита от leakage.

**Как понять, что sensitivity к prior — это проблема, а не свойство метода?**  
Зависимость evidence от prior принципиальна. Проблема возникает, если prior не указан, подстроен по test set или вывод меняется при малом произвольном изменении. Поэтому показываем sensitivity, а не пытаемся скрыть её.

**Почему нельзя всегда возвращать score с warning?**  
Если обязательный математический объект отсутствует, например posterior samples для WAIC, число будет не приближением метода, а другой величиной. В таких случаях правильнее structured failure, чем misleading score.

---

## Слайд 15. Summary

**Рекомендуемое время:** 35–40 секунд.

### Что сказать

«Итак, `dl_complexity` объединит классические criteria, MDL, Bayesian, prequential и
compression-based методы через общий контракт. Post-hoc оценки будут запускаться на
общих checkpoints, а методы с auxiliary inference или repeated training — на matched
experimental configurations. Их полезность мы сравним по связи с test NLL и
generalization gap, вычислительной стоимости и стабильности ranking.

Первый результат — end-to-end PoC на малых MLP; затем четыре ветки реализуются параллельно и проходят cross-review. Принципиальная часть результата — не только число, но и units, decomposition, assumptions и warnings. Ближайший технический шаг — утвердить типы `EstimationContext` и `ComplexityResult`, после чего реализовать BIC, two-part MDL и prequential PoC.»

### Как сформулировать главный результат одной фразой

> Мы не предлагаем ещё одну универсальную меру; мы создаём воспроизводимую систему, которая показывает, что именно измеряет каждая теоретически мотивированная оценка, сколько она стоит и насколько полезно ранжирует обученные модели по обобщению.

### Связка

«Последний слайд фиксирует основные источники, на которых основаны выбранные ветки.»

### Вероятные вопросы

**Что будет считаться научным результатом, кроме библиотеки?**  
Контролируемое эмпирическое сравнение: correlations и uncertainty, failure regimes, sensitivity к assumptions и cost–informativeness trade-off для разных estimator families.

**Какой результат опровергнет исходную мотивацию?**  
Если сложные estimates не превосходят parameter count или validation NLL по стабильности ranking и при этом значительно дороже, это тоже содержательный отрицательный результат.

**Какой estimator вы ожидаете увидеть лучшим?**  
Заранее победителя не фиксируем. Prequential score непосредственно измеряет learnability, Bayesian/MDL методы учитывают prior или code, но стоят дороже и чувствительны к approximation. Hypothesis должна проверяться benchmark-ом.

---

## Слайд 16. References

**Рекомендуемое время:** 20–25 секунд; при нехватке времени не перечислять вслух.

### Что сказать

«Для классических criteria мы опираемся на первичные работы Akaike, Schwarz,
Hannan и Quinn. Watanabe даёт теорию WAIC и WBIC для singular learning. Остальные
ветки основаны на tutorial Grünwald по MDL, работе Graves по variational inference,
Ritter, Botev и Barber по scalable Laplace approximation, Bornschein, Li и Hutter по
prequential MDL и работе Arora с соавторами по compression-based generalization bounds.
При реализации мы укажем точные theorem или section для каждого estimator-а.»

### Что важно знать о каждом источнике

**Akaike (1974), Schwarz (1978), Hannan и Quinn (1979)** вводят разные information
criteria с разными целями: predictive efficiency, regular-model evidence approximation
и consistent order selection соответственно. Их penalties нельзя считать
взаимозаменяемыми только из-за общей формы fit plus penalty.

**Watanabe (2010, 2013)** формализует WAIC как Bayesian predictive criterion и WBIC как
tempered-posterior approximation к Bayes free energy, в том числе в singular setting.
Сходство аббревиатур не означает, что WAIC и WBIC оценивают один объект.

**Grünwald (2004)** даёт введение в MDL, связь coding и learning, различия между two-part, refined/universal codes и model selection. Полезен, чтобы не отождествлять любой penalty с буквальным кодом.

**Graves (2011)** рассматривает variational inference для neural networks и практическую оптимизацию description length / variational free energy.

**Ritter, Botev, Barber (2018)** предлагает scalable Laplace approximation с Kronecker-factored structure для нейросетей.

**Bornschein, Li, Hutter (2022)** исследует sequential learning neural networks для prequential MDL и computationally efficient online variants.

**Arora et al. (2018)** связывает compressibility deep networks с stronger generalization bounds. Конкретные assumptions и форма bound должны цитироваться точно, а не заменяться схематичной формулой со слайда.

### Финальная фраза

«Спасибо. Мы готовы обсудить выбор estimator-ов, их предпосылки и экспериментальный протокол.»

### Вероятные вопросы

**Почему references по классическим criteria сгруппированы в одну строку?**  
Только для компактности слайда. В literature index и техническом отчёте каждая работа
получит полную библиографическую запись, а для каждого estimator-а будет указан
конкретный definition или theorem.

**Будет ли literature review частью отчёта?**  
Да, но структурированное по семантике методов: predictive criteria, marginal evidence approximations, explicit/sequential codes и compression bounds.

---

# Расширенный Q&A для защиты

## 1. Как соотносятся четыре семейства методов?

Короткий ответ: они отвечают на связанные, но не одинаковые вопросы.

| Семейство | Основной объект | Что считается «ценой сложности» | Главная оговорка |
|---|---|---|---|
| AIC и WAIC | ожидаемый predictive risk | аналитический/effective penalty к fit | асимптотические предпосылки и posterior quality |
| HQIC | consistent model-order selection | растущий penalty к fit | исходная гарантия не переносится на neural networks |
| BIC, WBIC, variational/Laplace | marginal likelihood / evidence | интегрирование по parameter space и prior | зависимость от prior и approximation |
| two-part/prequential MDL | длина decodable message | bits для модели и/или последовательных labels | нужно полностью задать код и protocol |
| compression bound | риск compressed hypothesis | description length в confidence penalty | bound относится к decoded compressed model |

Их объединяет принцип «хорошее объяснение данных должно учитывать fit и сложность», но нельзя утверждать, что все возвращают одно и то же число. Именно поэтому API сохраняет `native_value`, `units`, `components` и `assumptions`.

## 2. Что такое MDL простыми словами?

Minimum Description Length — принцип выбора объяснения, которое позволяет наиболее кратко передать данные. Если модель слишком проста, коротко описывается сама, но плохо кодирует данные. Если слишком сложна, хорошо fit-ит данные, но её описание дорого. Оптимум балансирует эти части.

Важное уточнение: существуют разные MDL constructions. Two-part MDL буквально передаёт модель, затем данные. Prequential MDL не передаёт параметры напрямую, а кодирует последовательные labels predictions обучающего алгоритма. Refined MDL использует universal distributions и не обязан сводиться к простой сумме «параметры + NLL».

## 3. Почему количество параметров недостаточно?

Parameter count не учитывает значения weights, симметрии, effective rank, flat directions, quantizability, prior/posterior concentration и способность learner-а быстро извлекать структуру. Сеть из миллиона почти нулевых или низкоранговых weights может быть эффективно проще другой сети той же формы. Но parameter count остаётся важным baseline: сложный estimator должен показать добавочную пользу.

## 4. Что такое singular model и почему это важно?

Regular statistical model предполагает локально взаимно однозначную параметризацию и невырожденную Fisher information около истинного параметра. Нейросети singular: перестановка hidden units, взаимная компенсация scale между layers и inactive units дают одинаковые функции для разных параметров; Fisher/Hessian имеют вырожденные направления. Поэтому classical Laplace/BIC asymptotics с penalty $k\log n$ могут быть неверны. WAIC/WBIC мотивированы singular learning theory, но их практическая оценка всё равно требует качественного posterior inference.

## 5. Что такое evidence и Occam factor?

Evidence:

\[
p(D\mid M)=\int p(D\mid\theta,M)p(\theta\mid M)\,d\theta.
\]

Он усредняет fit по всему prior parameter space. Гибкая модель может иметь очень высокий maximum likelihood, но если хорошая область занимает крошечный prior volume, average evidence будет умеренным. Отношение posterior volume к prior volume часто называют Occam factor. Это не «штраф, добавленный вручную», а эффект интегрирования. Однако evidence принципиально зависит от prior.

## 6. Почему prior нельзя назвать субъективной помехой и убрать?

Evidence, KL и Bayesian description length без prior не определены. Prior задаёт исходную precision и систему отсчёта информации. Его можно выбирать иерархически или на training data по заранее заданной процедуре, но нельзя оптимизировать по held-out test. Следует показать sensitivity к разумному диапазону prior scales.

## 7. В чём разница между likelihood, log-likelihood, NLL и cross-entropy?

- Likelihood: $p(D\mid\theta)$, рассматриваемый как функция параметров.
- Log-likelihood: $\ell(\theta)=\sum_i\log p(y_i\mid x_i,\theta)$.
- NLL: $-\ell(\theta)$; бывает суммой или средним, это нужно указывать.
- Cross-entropy: для one-hot labels и softmax probabilities совпадает со средней или суммарной NLL в зависимости от reduction.

В формулах AIC/BIC нужна **суммарная** log-likelihood на training data. Подстановка средней NLL без умножения на $n$ радикально исказит отношение fit и penalty.

## 8. Почему у AIC/BIC коэффициент 2?

Исторически критерии задаются в deviance scale $-2\log L$. Это удобно для likelihood-ratio asymptotics. Если логарифм натуральный, половина критерия измеряется в nats-like scale. Коэффициент 2 не несёт самостоятельного информационного содержания для ranking, но важен при сравнении компонентов и переводе units.

## 9. Как считать WAIC численно устойчиво?

Имея matrix pointwise log-likelihood $\ell_{si}=\log p(y_i\mid x_i,\theta^{(s)})$, считать:

\[
\mathrm{lppd}_i=\log\frac1S\sum_s e^{\ell_{si}}
=\operatorname{logsumexp}_s(\ell_{si})-\log S,
\]

а $p_{\mathrm{WAIC},i}$ — sample variance $\ell_{si}$ по $s$, с явно выбранным convention для denominator. Затем суммировать по $i$. Нельзя сначала усреднить log-likelihood и затем взять exponent: из-за неравенства Йенсена это другая величина.

## 10. WAIC и WBIC — одно и то же?

Нет. WAIC оценивает Bayesian predictive performance, используя posterior predictive density и effective parameter count. WBIC оценивает Bayes free energy / negative log evidence через expectation при tempered posterior. Названия похожи, но цели, formula и необходимые samples различаются.

## 11. Что означает температура в WBIC?

В power posterior likelihood возводится в степень $\beta$:

\[
p_\beta(\theta\mid D)\propto p(D\mid\theta)^\beta p(\theta).
\]

При $\beta=0$ это prior, при $\beta=1$ обычный posterior. WBIC использует $\beta=1/\log n$, то есть более горячее и широкое распределение. Samples обычного posterior не являются samples WBIC без корректного reweighting, которое само может быть нестабильным.

## 12. Что именно утверждает ELBO identity?

\[
\log p(D)
=\mathrm{ELBO}(q)
+\mathrm{KL}(q\|p(\theta\mid D)).
\]

Поскольку второй член неотрицателен, ELBO не превосходит log evidence. Равенство достигается только при $q=p(\theta\mid D)$. Поэтому улучшение ELBO может означать и лучший data fit, и приближение posterior; смотреть только итог без decomposition недостаточно.

## 13. Почему направление KL именно $\mathrm{KL}(q\|p)$?

Стандартный variational inference минимизирует $\mathrm{KL}(q(\theta)\|p(\theta\mid D))$, что получается из ELBO identity. В negative ELBO complexity term равен $\mathrm{KL}(q(\theta)\|p(\theta))$ к prior. Это два разных KL. Reverse KL обычно mode-seeking: ограниченное семейство $q$ может покрыть один mode и недооценить posterior uncertainty.

## 14. Что такое bits-back и почему нужна оговорка?

Наивный код latent $\theta\sim q$ стоил бы $-\log p(\theta)$, но bits-back scheme позволяет «вернуть» примерно $-\log q(\theta)$ bits, давая среднюю цену $\mathrm{KL}(q\|p)$ плюс data cost. Для буквальной реализации нужны общий source of initial bits, дискретизация/finite precision и entropy coder. Поэтому negative ELBO имеет идеализированную кодовую интерпретацию, но без encoder/decoder это variational bound, а не измеренный размер сообщения.

## 15. Что такое Hessian, Fisher и GGN?

- **Hessian** — matrix вторых производных выбранного scalar objective по parameters.
- **Fisher information** — expectation outer product score gradients или negative expected Hessian log-likelihood при regular conditions.
- **Generalized Gauss–Newton (GGN)** — PSD approximation curvature для compositional models с convex loss по outputs.

Они могут совпадать в специальных условиях, но в общем различаются. KFAC — способ факторизованно приблизить block structure, часто empirical Fisher или GGN. Поэтому в report нужно точно назвать, какая matrix использована под символом $H$.

## 16. Что означает log determinant в Laplace approximation?

Если $H$ положительно определена с eigenvalues $\lambda_1,\ldots,\lambda_d$, то

\[
\log\det H=\sum_{r=1}^{d}\log\lambda_r.
\]

Это мера inverse local volume posterior ellipsoid. Большая curvature сжимает posterior volume и увеличивает negative log evidence correction. Вычислять determinant напрямую нельзя: он переполняется/исчезает; используют Cholesky или сумму log eigenvalues.

## 17. Почему damping меняет статистический смысл?

Замена $H$ на $H+\lambda I$ меняет covariance и log determinant. Если $\lambda$ соответствует prior precision, это часть probabilistic model. Если добавлена исключительно для численной устойчивости, результат всё равно меняется; поэтому значение $\lambda$ сообщается и варьируется в sensitivity analysis.

## 18. Почему flat minimum не всегда означает хорошее обобщение?

Sharpness зависит от параметризации: rescaling layers может изменить Hessian eigenvalues, сохранив функцию. Prior-aware evidence частично задаёт scale, но простая curvature metric не является полностью invariant. Поэтому мы не утверждаем «flatness causes generalization» и оцениваем Laplace score эмпирически с явной parameterization.

## 19. Почему prequential code не передаёт weights?

Encoder и decoder запускают заранее согласованный deterministic learner на уже переданных данных и получают одну и ту же модель. Поэтому достаточно передавать labels следующего блока с её predictive probabilities. Цена model learning проявляется косвенно: learner, который на малом prefix делает плохие predictions, расходует больше bits.

## 20. Как block schedule влияет на prequential length?

Большой первый блок дорог, потому что кодируется uniform. Слишком много маленьких блоков требуют много fits и могут давать другой training trajectory. Поэтому schedule — часть кода, задаётся до наблюдения test results и включается в metadata. Сравнивать методы нужно при одном schedule или показывать sensitivity.

## 21. Является ли $-\log p$ настоящим числом битов?

Это идеальная information content. Arithmetic/range coding может приблизиться к суммарной cross-entropy с небольшим overhead, но требует finite-precision probabilities и согласованного coder state. Для теоретической оценки допустима fractional bit length; для заявления о round-trip compressor нужен реальный encoder/decoder.

## 22. Что именно нужно закодировать при compression?

Всё, чего decoder заранее не знает: architecture при необходимости, tensor shapes, pruning masks или indices, codebooks, quantized values, ranks, factors, precisions и side information. Общие алгоритмы encoder/decoder не оплачиваются на каждый объект, если они заранее согласованы, но data-dependent choices должны быть отражены в сообщении или покрыты theorem.

## 23. Что такое generalization bound и чем он отличается от estimate?

Bound — вероятностное верхнее ограничение на population risk при чётких assumptions и confidence $1-\delta$. Estimate — числовой surrogate без такой гарантии. Bound может быть математически корректным, но vacuous, то есть выше максимального возможного риска. Поэтому нужно отдельно проверять validity и tightness.

## 24. Можно ли выбирать compression level, минимизирующий bound?

Если несколько levels проверяются на тех же training data, выбор должен быть покрыт union bound, prior over codes или дополнительной стоимостью выбора. Выбирать level по test set нельзя. Практически мы фиксируем grid заранее и либо показываем всю rate–distortion curve, либо учитываем selection correction.

## 25. Почему test set нельзя использовать даже для tuning estimator-а?

Цель проекта — измерить связь score с неизвестным будущим качеством. Если prior, damping, blocks или precision выбираются для максимальной test correlation, test information уже встроена в score, и reported correlation оптимистична. Настройка идёт на validation/meta-development split или из теории; окончательный test открывается после freeze.

## 26. Почему validation NLL является baseline, если она уже хорошо выбирает модели?

Именно поэтому. Практическая ценность complexity estimator должна сравниваться с простым способом model selection. Возможные преимущества complexity methods — работа при малом validation set, структурная интерпретация, связь с theory или robustness across shifts — но это нужно продемонстрировать, а не предположить.

## 27. Как трактовать корреляцию с generalization gap?

Положительная $\rho_s$ означает, что модели с большим score обычно имеют больший NLL gap. Но gap содержит train NLL, а многие scores тоже содержат train fit; поэтому возможна алгебраическая зависимость. Нужно дополнительно показывать correlation с test NLL, penalty-only components и stratified analyses при близком train fit.

## 28. Почему нельзя ранжировать по абсолютным значениям всех методов в одной таблице?

Можно вывести все значения, но нельзя считать их одной общей cardinal scale. Различаются units, additive constants, conditioning и semantic target. Ranking внутри метода по одинаковому protocol обычно осмысленнее. Cross-method comparison относится к predictive usefulness, computational cost и stability, а не к тому, у кого «меньше число».

## 29. Какие статистические ошибки возможны в benchmark?

- слишком мало независимых model configurations;
- pseudo-replication, когда seeds считаются независимыми задачами;
- pooling разных datasets и sizes;
- selection лучшего estimator hyperparameter по test correlation;
- reporting только point estimate без confidence interval;
- multiple comparisons без оговорки;
- shared test set, превращённый итеративной разработкой в validation.

Минимальная защита: preregistered grid, held-out final split, paired runs, per-stratum results, bootstrap/seed uncertainty и публикация всех configs.

## 30. Что делать, если разные seeds меняют ranking?

Это не просто шум, а результат: estimator нестабилен относительно training randomness. Нужно сообщить distribution ranks/correlations, Kendall/Spearman agreement между seeds и, возможно, усреднять score на уровне configuration. Скрывать нестабильность выбором удобного seed нельзя.

## 31. Как проверять корректность реализации формул?

Использовать несколько уровней:

1. toy analytical cases, например Gaussian/linear/logistic models;
2. ручные calculations на двух–трёх точках;
3. comparison с trusted library там, где возможно;
4. invariance/metamorphic tests: batch partition не меняет summed likelihood, bits/nats conversion обратим;
5. failure tests: missing samples, non-positive curvature, invalid splits;
6. end-to-end reproducibility.

## 32. Почему проект называется библиотекой, а не только исследовательским notebook?

Единый typed API, validation, serialization и tests делают assumptions явными и позволяют повторить сравнение. Notebook остаётся presentation/demo layer, но не хранит скрытую business logic. Это снижает вероятность незаметно реализовать один и тот же термин по-разному в разных экспериментах.

---

# Краткий справочник формул

## Classical criteria

\[
\mathrm{AIC}=-2\log\hat L+2k,
\qquad
\mathrm{BIC}=-2\log\hat L+k\log n,
\]

\[
\mathrm{HQIC}=-2\log\hat L+2k\log\log n.
\]

Здесь $\hat L=p(D\mid\hat\theta)$, $k$ — число оценённых свободных параметров, а
$n$ — число независимых observations. Effective dimension допустима только как
отдельно названная модификация стандартного критерия.

## WAIC

\[
\mathrm{lppd}=\sum_i\log\mathbb E_{p(\theta\mid D)}
[p(y_i\mid x_i,\theta)],
\]

\[
p_{\mathrm{WAIC}}=\sum_i\operatorname{Var}_{p(\theta\mid D)}
[\log p(y_i\mid x_i,\theta)],
\]

\[
\mathrm{WAIC}=-2(\mathrm{lppd}-p_{\mathrm{WAIC}}).
\]

## WBIC

\[
p_\beta(\theta\mid D)\propto p(D\mid\theta)^\beta p(\theta),
\quad \beta=1/\log n,
\]

\[
\mathrm{WBIC}=\mathbb E_{p_\beta}[-\log p(D\mid\theta)].
\]

## Variational objective

\[
-\mathrm{ELBO}
=\mathbb E_q[-\log p(D\mid\theta)]
+\mathrm{KL}(q\|p),
\]

\[
-\log p(D)
=-\mathrm{ELBO}
-\mathrm{KL}(q\|p(\theta\mid D)).
\]

Следовательно, $-\mathrm{ELBO}\ge-\log p(D)$.

## Laplace evidence

\[
H=\nabla^2_\theta[-\log p(D,\theta)]_{\hat\theta},
\]

\[
-\log p(D)\approx
-\log p(D,\hat\theta)
+\frac12\log\det H-\frac d2\log(2\pi).
\]

## Prequential code

\[
L_{\mathrm{preq}}=n_1\log_2K
-\sum_{j=2}^m\sum_{i=n_{j-1}+1}^{n_j}
\log_2p_{\hat\theta(D_{<j})}(y_i\mid x_i).
\]

## Compression-bound template

\[
R(C)\leq \widehat R(C)
+\sqrt{\frac{\ell(C)\ln 2+\ln(1/\delta)}{2n}}.
\]

Это template для bounded loss и prefix-coded hypothesis. Для численного bound нужна
конкретная theorem с constants, loss assumptions и согласованными единицами.

## Benchmark targets

\[
\overline{\mathrm{NLL}}_S
=-\frac1{|S|}\sum_{i\in S}\ln p(y_i\mid x_i,\hat\theta),
\qquad
g_{\mathrm{NLL}}
=\overline{\mathrm{NLL}}_{\mathrm{test}}
-\overline{\mathrm{NLL}}_{\mathrm{train}}.
\]

\[
\rho_s=\operatorname{Corr}_{\mathrm{Pearson}}(R(c),R(t)).
\]

---

# Glossary

**AIC** — Akaike Information Criterion; predictive information criterion с penalty $2k$.

**Assumption** — условие, при котором определение, approximation или theorem применимы.

**Bayes evidence / marginal likelihood** — вероятность данных после интегрирования parameters по prior.

**BIC** — Bayesian/Schwarz Information Criterion; regular asymptotic approximation к $-2\log$ evidence.

**Bits** — единицы information content при $\log_2$.

**Bits-back coding** — схема, позволяющая получить variational code length, близкую к negative ELBO, за счёт возврата entropy $q$.

**Calibration** — согласованность predicted probabilities с empirical frequencies; влияет на NLL и predictive code length.

**Checkpoint** — сохранённые weights и связанные артефакты конкретного training run.

**Code length** — длина однозначно декодируемого сообщения; в идеализированном probability code равна $-\log p$.

**Compression ratio** — отношение исходного размера представления к encoded size; definition должна включать units и side information.

**Curvature** — локальная второпроизводная информация objective по parameters.

**Damping** — стабилизирующий positive shift curvature matrix; влияет на covariance и evidence.

**ELBO** — Evidence Lower Bound, variational lower bound на log evidence.

**Estimator** — программный объект, вычисляющий конкретный complexity criterion/estimate из context.

**Generalization gap** — разность held-out и training quality в одной и той же metric.

**Generalization bound** — вероятностная theorem, ограничивающая population risk через empirical quantities и complexity.

**GGN** — generalized Gauss–Newton matrix, PSD curvature approximation.

**Hessian** — matrix вторых partial derivatives scalar objective.

**HQIC** — Hannan–Quinn Information Criterion с penalty $2k\log\log n$.

**Identifiability** — свойство, при котором разные parameters задают разные distributions, кроме допустимых эквивалентностей.

**KFAC** — Kronecker-Factored Approximate Curvature; layerwise Kronecker approximation Fisher/GGN structure.

**KL divergence** — несимметричная мера расхождения distributions, $\mathbb E_q[\log(q/p)]$.

**Likelihood** — вероятность наблюдённых данных как функция parameters.

**lppd** — log pointwise predictive density; сумма логарифмов posterior-усреднённых pointwise probabilities.

**MAP** — maximum a posteriori estimate, maximizer posterior или joint density.

**MDL** — Minimum Description Length, принцип выбора кратчайшего полного описания модели/данных.

**MLE** — maximum likelihood estimate.

**Nats** — единицы information content при натуральном logarithm; $1\text{ nat}=1/\ln2\text{ bits}$.

**NLL** — negative log-likelihood; меньше означает лучший probabilistic fit.

**Occam factor** — evidence penalty за малый posterior-compatible volume относительно prior volume.

**Pointwise log-likelihood** — отдельное $\log p(y_i\mid x_i,\theta)$ для каждого observation.

**Posterior** — distribution parameters после учёта данных: $p(\theta\mid D)$.

**Prefix code** — однозначно декодируемый code, ни одно codeword которого не является prefix другого.

**Prequential** — predictive sequential; данные кодируются predictions model, обученной только на прошлом prefix.

**Prior** — distribution parameters до текущих данных; обязательная часть Bayesian score.

**Quantization** — отображение continuous weights на конечный набор levels для конечной кодовой длины.

**Regular model** — модель с локальной identifiability и невырожденной information geometry.

**Round-trip test** — encode → decode с проверкой точного восстановления заявленного представления.

**Singular model** — модель с неидентифицируемостью или вырожденной Fisher/Hessian geometry.

**Spearman correlation** — Pearson correlation ranks; измеряет силу монотонной связи.

**Tempered posterior** — posterior с likelihood в степени inverse temperature $\beta$.

**Two-part MDL** — код, отдельно передающий model representation и затем данные conditional on model.

**WAIC** — Widely Applicable Information Criterion; Bayesian predictive criterion по posterior pointwise likelihood.

**WBIC** — Widely Applicable Bayesian Information Criterion; tempered-posterior approximation к Bayes free energy.

---

# Последняя памятка перед выступлением

Перед защитой убедитесь, что вы можете без слайда ответить на шесть вопросов:

1. Почему parameter count — только baseline?
2. Чем predictive criterion отличается от evidence и literal code?
3. Почему WAIC/WBIC нельзя считать из одного checkpoint?
4. Почему KFAC–Laplace содержит несколько уровней approximation?
5. Как decoder воспроизводит prequential code без future-label leakage?
6. Почему compression bound относится к compressed model и почему test set не используется для tuning?

Если формулу забыли, сначала объясните объект словами, затем назовите компоненты и assumptions. Лучше честно сказать «на слайде показана общая форма, точные constants зависят от выбранной theorem», чем приписать схематической формуле строгую гарантию.
