# Литература проекта

Здесь собраны первоисточники из задания преподавателя. PDF имеют короткие стабильные
имена; новые материалы лучше добавлять по схеме `author_year_short_title.pdf` и сразу
описывать в этой таблице.

| Файл | Источник | Для чего нужен |
| --- | --- | --- |
| [`grunwald_2004_mdl_tutorial.pdf`](grunwald_2004_mdl_tutorial.pdf) | P. Grünwald, *A Tutorial Introduction to the Minimum Description Length Principle* ([arXiv](https://arxiv.org/abs/math/0406077)) | Основа MDL и 2-part codes; особенно §2.6.3 для связи байесовского вывода и MDL. |
| [`graves_2011_variational_inference.pdf`](graves_2011_variational_inference.pdf) | A. Graves, *Practical Variational Inference for Neural Networks* ([NeurIPS](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)) | Вариационная objective как MDL-критерий для нейросетей. |
| [`ritter_2018_scalable_laplace.pdf`](ritter_2018_scalable_laplace.pdf) | H. Ritter, A. Botev, D. Barber, *A Scalable Laplace Approximation for Neural Networks* ([OpenReview](https://openreview.net/forum?id=Skdvd2xAZ), [UCL mirror](https://discovery.ucl.ac.uk/id/eprint/10080902/)) | Evidence через Laplace approximation и KFAC-структуру кривизны. |
| [`bornschein_2022_prequential_mdl.pdf`](bornschein_2022_prequential_mdl.pdf) | J. Bornschein, Y. Li, M. Hutter, *Sequential Learning of Neural Networks for Prequential MDL* ([arXiv](https://arxiv.org/abs/2210.07931)) | Blockwise/online prequential code, rehearsal и calibration. |
| [`arora_2018_compression_generalization.pdf`](arora_2018_compression_generalization.pdf) | S. Arora et al., *Stronger Generalization Bounds for Deep Nets via a Compression Approach* ([PMLR](https://proceedings.mlr.press/v80/arora18b.html)) | Компрессия сети и основанная на ней граница обобщения. |

## Справочные темы без единственного обязательного PDF

- AIC, BIC и HQIC: определения, соглашения о знаке log-likelihood и подсчёт эффективного
  числа параметров должны быть зафиксированы до реализации.
- WAIC: нужны posterior samples и pointwise log-likelihood; это не просто функция от
  одной обученной сети.
- WBIC: требует ожидания log-likelihood при tempered posterior, обычно с
  `beta = 1 / log(n)`.
- Длины следует хранить в одной базовой единице; перевод: `bits = nats / ln(2)`.

## Как добавить свой PDF

1. Положить файл в эту папку, не меняя и не коммитя чужие лицензионные материалы.
2. Добавить строку с авторами, постоянной ссылкой и ролью статьи в проекте.
3. Для новой формулы указать страницу/раздел и используемые в реализации допущения.

Все автоматически добавленные статьи доступны в открытых научных архивах. Права на
тексты принадлежат их авторам и издателям; файлы используются для учебной работы.
