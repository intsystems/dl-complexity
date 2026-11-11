# Справочник публичного API

_DL Complexity 0.1.0a1. Импорты, параметры вызовов и возвращаемые объекты._

---

Оценщики используют общий интерфейс `check` / `estimate`. Подготовка posterior, обучение и сжатие вызываются отдельно. Примеры и формулы приведены в [руководстве](CODE_WALKTHROUGH.md), внутренняя структура — в [карте исходников](../src/dl_complexity/README.md).

Конструкторы записей перечисляют все поля. Обозначение `<factory>` означает, что для каждого экземпляра создаётся новый список или словарь. Имена типов в кавычках — отложенные аннотации Python. Блоки сигнатур служат справкой, а не самостоятельными программами.

Параметры и ограничения в раскрывающихся блоках даны на языке docstrings — английском. Унаследованные `check`, `estimate` и `config` описаны один раз у [ComplexityEstimator](#api-dl_complexity-base-complexityestimator). Внутренние методы с одиночным начальным `_` не относятся к пользовательскому API.

## 📚 Разделы

- [🧩 Общие объекты](#reference-core)
- [📊 Информационные критерии](#reference-criteria)
- [🧮 Байесовские оценки](#reference-bayesian)
- [📦 Кодовые длины](#reference-coding)
- [🗜️ Преобразования и кодеки](#reference-compression)
- [📐 Границы риска](#reference-bounds)
- [📏 Диагностики весов](#reference-diagnostics)
- [🔌 PyTorch-интеграции](#reference-integrations)
- [⚠️ Исключения](#reference-errors)

<a id="reference-core"></a>

## 🧩 Общие объекты

Импорт: `from dl_complexity import ...`.

[ComplexityEstimator](#api-dl_complexity-base-complexityestimator) | [ComplexityResult](#api-dl_complexity-results-complexityresult) | [ComplexitySuite](#api-dl_complexity-evaluation-complexitysuite) | [DataSource](#api-dl_complexity-data-datasource) | [EvaluationOutcome](#api-dl_complexity-evaluation-evaluationoutcome) | [ModelContext](#api-dl_complexity-context-modelcontext) | [ModelStructure](#api-dl_complexity-context-modelstructure) | [Provenance](#api-dl_complexity-results-provenance) | [Quantity](#api-dl_complexity-results-quantity) | [RequirementReport](#api-dl_complexity-results-requirementreport) | [RequirementStatus](#api-dl_complexity-results-requirementstatus) | [TrainingRecipe](#api-dl_complexity-context-trainingrecipe) | [TrainingRecord](#api-dl_complexity-context-trainingrecord)

<a id="api-dl_complexity-base-complexityestimator"></a>

### `ComplexityEstimator`

[Исходник](../src/dl_complexity/base.py#L16)

```text
ComplexityEstimator(**config)
```

Define the shared typed evaluation protocol, without an obligatory fit.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    **config: Constructor settings recorded in every result.

Attributes:
    input_type (type or tuple[type]): Accepted prepared-input record classes.
    uses_data (bool): Whether the context route may consume a data source.

Notes:
    Subclasses implement ``_estimate`` and optionally a direct context route.
    Posterior sampling, optimization and curvature collection remain explicit
    preparation functions, never accidental effects of ``check``.
```

</details>

---

<a id="api-dl_complexity-base-complexityestimator-config"></a>

#### `ComplexityEstimator.config`

[Исходник](../src/dl_complexity/base.py#L46)

Свойство, читается без вызова.

Return a shallow copy of constructor configuration.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    dict: Independent key mapping; nested values may still be shared.
```

</details>

---

<a id="api-dl_complexity-base-complexityestimator-check"></a>

#### `ComplexityEstimator.check`

[Исходник](../src/dl_complexity/base.py#L55)

```text
ComplexityEstimator.check(context=None, *, data=None, inputs=None) -> 'RequirementReport'
```

Check input metadata and declared identities without executing a model.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext, optional): Model and likelihood declarations.
    data (DataSource, optional): Observation source, not consumed here.
    inputs (object, optional): This method's prepared numerical record.

Returns:
    RequirementReport: Computability, missing information and assumptions.

Raises:
    InputError: Wrong input type or conflicting nonempty provenance IDs.

Notes:
    Dispatches to family metadata checks. READY is not evidence of MLE/MAP,
    iid sampling or convergence; unknown identities are not content hashes.
    No forward, training, sampling, RNG mutation or iterator consumption.
```

</details>

---

<a id="api-dl_complexity-base-complexityestimator-estimate"></a>

#### `ComplexityEstimator.estimate`

[Исходник](../src/dl_complexity/base.py#L139)

```text
ComplexityEstimator.estimate(context=None, *, data=None, inputs=None) -> 'ComplexityResult'
```

Validate, resolve direct inputs and evaluate one mathematical quantity.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext, optional): Ordinary PyTorch model wrapper.
    data (DataSource, optional): Data for a supported direct evaluation route.
    inputs (object, optional): Sufficient prepared values; bypass model forward.

Returns:
    ComplexityResult: Named quantities, settings, provenance and elapsed time.

Raises:
    InputError: Invalid or missing mathematical information.
    PreparationRequired: A declared preparation action must be called first.

Notes:
    Calls check, optionally _prepare_context, then _estimate. Result-specific
    configuration overrides duplicate constructor keys. Time excludes preflight
    and is a host perf_counter interval, not a synchronized GPU benchmark.
    Wrapper cost is constant apart from copied configuration and family work.
```

</details>

---

<a id="api-dl_complexity-results-complexityresult"></a>

### `ComplexityResult`

[Исходник](../src/dl_complexity/results.py#L115)

```text
ComplexityResult(
    method: 'str',
    kind: 'str',
    quantities: 'Mapping[str, Quantity]',
    primary_name: 'str | None' = None,
    assumptions: 'tuple[str, ...]' = (),
    config: 'Mapping[str, Any]' = <factory>,
    provenance: 'Provenance | None' = None,
    elapsed_seconds: 'float | None' = None
)
```

Return named mathematical outputs and their interpretation.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    method (str): Estimator name.
    kind (str): Estimand, distinct from units or module family.
    quantities (Mapping[str, Quantity]): Primary and component quantities.
    primary_name (str, optional): Primary key; diagnostics can omit it.
    assumptions (tuple[str, ...]): Conditions, approximations and limitations.
    config (Mapping): Constructor settings plus actual calculation metadata.
    provenance (Provenance, optional): Declared input identities.
    elapsed_seconds (float, optional): Host-side interval, not GPU benchmark.

Notes:
    Mappings/tensors are not recursively frozen. No cross-kind ranking is defined.
```

</details>

---

<a id="api-dl_complexity-results-complexityresult-primary"></a>

#### `ComplexityResult.primary`

[Исходник](../src/dl_complexity/results.py#L157)

Свойство, читается без вызова.

Look up the named primary quantity, or return None for multi-output diagnostics.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    Quantity or None: Existing quantity; no tensor is copied.
```

</details>

---

<a id="api-dl_complexity-evaluation-complexitysuite"></a>

### `ComplexitySuite`

[Исходник](../src/dl_complexity/evaluation.py#L32)

```text
ComplexitySuite(methods: 'Mapping[str, ComplexityEstimator]', *, errors='raise')
```

Compose named estimators while preserving heterogeneous result meanings.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    methods (Mapping[str, ComplexityEstimator]): Explicit method selection.
    errors (str): 'raise' by default, or 'collect' for per-name outcomes.

Notes:
    No training scheduler, parallel backend, automatic ranking or global cache.
```

</details>

---

<a id="api-dl_complexity-evaluation-complexitysuite-check"></a>

#### `ComplexitySuite.check`

[Исходник](../src/dl_complexity/evaluation.py#L77)

```text
ComplexitySuite.check(context=None, *, data=None, inputs=None)
```

Run every method's metadata preflight without consuming shared data.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext, optional): Shared model wrapper.
    data (DataSource, optional): Shared source metadata.
    inputs (Mapping, optional): Name to typed prepared input.

Returns:
    dict[str, RequirementReport]: Report for every selected method.

Notes:
    In collect mode InputError/TypeError become UNSUPPORTED reports. Unknown
    input names still fail. Cost is the sum of family checks, not model forward.
```

</details>

---

<a id="api-dl_complexity-evaluation-complexitysuite-estimate"></a>

#### `ComplexitySuite.estimate`

[Исходник](../src/dl_complexity/evaluation.py#L108)

```text
ComplexitySuite.estimate(context=None, *, data=None, inputs=None)
```

Preflight all methods, then evaluate selected methods sequentially.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext, optional): Shared context.
    data (DataSource, optional): Reusable source when multiple readers need it.
    inputs (Mapping, optional): Prepared inputs indexed by method name.

Returns:
    dict[str, EvaluationOutcome]: One named outcome, including missing inputs.

Raises:
    InputError: Failed strict preflight or multiple readers of one-shot data.

Notes:
    Cost is the sum of method evaluations; shared contexts do not automatically
    share likelihood passes. Reuse one FitStatistics explicitly to avoid them.
    Returned results may keep autograd graphs alive. Collect mode catches
    ordinary Exceptions; strict execution has no transactional rollback.
```

</details>

---

<a id="api-dl_complexity-data-datasource"></a>

### `DataSource`

[Исходник](../src/dl_complexity/data.py#L14)

```text
DataSource(
    batches: 'Iterable | Callable[[], Iterable]',
    n_observations: 'int | None' = None,
    data_id: 'str | None' = None
)
```

Describe observation batches without silently downloading or materializing them.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    batches (Iterable or Callable): Batches (inputs, targets), or a fresh factory.
    n_observations (int, optional): Declared count, validated after consumption.
    data_id (str, optional): Declared dataset/preprocessing identity.

Notes:
    Targets' first axis indexes observations; likelihood sums event axes.
    A one-shot iterator is permitted for one explicit evaluation, not shared
    consumption by several suite readers. Construction does not iterate.
```

</details>

---

<a id="api-dl_complexity-data-datasource-from_tensors"></a>

#### `DataSource.from_tensors`

[Исходник](../src/dl_complexity/data.py#L38)

```text
DataSource.from_tensors(inputs: 'torch.Tensor', targets: 'torch.Tensor', *, data_id=None)
```

Wrap tensors as one reusable full batch without copying their values.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    inputs (torch.Tensor): Shape [N, ...], sharing first axis with targets.
    targets (torch.Tensor): Shape [N, ...] with N positive.
    data_id (str, optional): Dataset identity.

Returns:
    DataSource: Single-batch iterable retaining references to the input tensors.

Raises:
    InputError: Missing observation axis or inconsistent lengths.

Notes:
    Metadata-only O(1) wrapping, but the caller already owns the full dataset.
```

</details>

---

<a id="api-dl_complexity-data-datasource-__iter__"></a>

#### `DataSource.__iter__`

[Исходник](../src/dl_complexity/data.py#L62)

```text
DataSource.__iter__()
```

Create a batch iterator, calling the factory only at explicit iteration time.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    Iterator: Underlying iterator; arbitrary factory costs belong to the caller.
```

</details>

---

<a id="api-dl_complexity-data-datasource-one_shot"></a>

#### `DataSource.one_shot`

[Исходник](../src/dl_complexity/data.py#L70)

Свойство, читается без вызова.

Detect an existing iterator by __next__ without invoking a factory or __iter__.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    bool: Structural reuse warning, not a proof that a custom iterable is repeatable.
```

</details>

---

<a id="api-dl_complexity-evaluation-evaluationoutcome"></a>

### `EvaluationOutcome`

[Исходник](../src/dl_complexity/evaluation.py#L14)

```text
EvaluationOutcome(
    result: 'ComplexityResult | None' = None,
    requirements: 'RequirementReport | None' = None,
    error: 'Exception | None' = None
)
```

Retain a result, preflight report or ordinary runtime error for one name.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    result (ComplexityResult, optional): Successful evaluation.
    requirements (RequirementReport, optional): Preflight status.
    error (Exception, optional): Failure retained when errors='collect'.

Notes:
    An unavailable score is not replaced by zero or silently dropped.
```

</details>

---

<a id="api-dl_complexity-context-modelcontext"></a>

### `ModelContext`

[Исходник](../src/dl_complexity/context.py#L87)

```text
ModelContext(
    model: 'nn.Module',
    loss: 'Callable | None' = None,
    likelihood: 'Any | None' = None,
    training: 'TrainingRecipe | None' = None,
    state_id: 'str | None' = None,
    likelihood_id: 'str | None' = None,
    loss_likelihood_relation: 'str' = 'unspecified',
    runs: 'list[TrainingRecord]' = <factory>
)
```

Compose an ordinary nn.Module with likelihood, loss and training metadata.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Existing model, not copied by this constructor.
    loss (Callable, optional): User loss; never assumed to define a density.
    likelihood (object, optional): Callable or object exposing log_prob(pred, y).
    training (TrainingRecipe, optional): Recipe, not an automatic fit request.
    state_id (str, optional): Declared model identity, not a weight checksum.
    likelihood_id (str, optional): Declared likelihood identity.
    loss_likelihood_relation (str): Description, not a verified algebraic mapping.
    runs (list[TrainingRecord]): Metadata of explicit run_training calls.

Notes:
    Constructing the wrapper does not train, trace or inspect parameter values.
```

</details>

---

<a id="api-dl_complexity-context-modelcontext-structure"></a>

#### `ModelContext.structure`

[Исходник](../src/dl_complexity/context.py#L119)

Свойство, читается без вызова.

Recompute module/parameter metadata and tied-object aliases.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    ModelStructure: Named modules and every named parameter reference.

Notes:
    Uses id(parameter), not storage overlap, for aliases. Time/memory scale
    with module and parameter-reference counts, not the number of weight values.
```

</details>

---

<a id="api-dl_complexity-context-modelcontext-from_checkpoint"></a>

#### `ModelContext.from_checkpoint`

[Исходник](../src/dl_complexity/context.py#L147)

```text
ModelContext.from_checkpoint(
    path: 'str | Path',
    *,
    model_factory: 'Callable[[], nn.Module]',
    map_location='cpu',
    **kwargs
)
```

Load a plain weights-only state_dict using a caller-supplied architecture.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    path (str or Path): Ordinary state_dict checkpoint, not a pickled module.
    model_factory (Callable): Builds matching architecture and parameter dtypes.
    map_location: Passed to torch.load; factory controls destination modules.
    **kwargs: ModelContext fields such as likelihood and declared state_id.

Returns:
    ModelContext: Fresh model with strict-loaded checkpoint tensors.

Raises:
    InputError: Not a state mapping or factory dtype would silently cast values.
    RuntimeError: PyTorch rejects missing, extra or shape-mismatched state keys.

Notes:
    Deserialize/copy cost is proportional to checkpoint bytes, in addition to
    model_factory work. An architecture cannot be recovered from state_dict alone.
    weights_only reduces pickle exposure; do not treat untrusted files as safe.
```

</details>

---

<a id="api-dl_complexity-context-modelstructure"></a>

### `ModelStructure`

[Исходник](../src/dl_complexity/context.py#L71)

```text
ModelStructure(
    modules: 'tuple[tuple[str, str], ...]',
    parameters: 'tuple[ParameterInfo, ...]'
)
```

Represent named modules and parameter references, not a forward graph.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    modules (tuple): Pairs of module path and class name.
    parameters (tuple[ParameterInfo, ...]): Parameter-reference metadata.

Notes:
    No forward execution, FX/export tracing or persistent autograd graph.
```

</details>

---

<a id="api-dl_complexity-results-provenance"></a>

### `Provenance`

[Исходник](../src/dl_complexity/results.py#L16)

```text
Provenance(
    state_id: 'str | None' = None,
    data_id: 'str | None' = None,
    likelihood_id: 'str | None' = None
)
```

Declare optional model, dataset and likelihood identities.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    state_id (str, optional): Identity for a fixed model state.
    data_id (str, optional): Identity for observations/preprocessing.
    likelihood_id (str, optional): Identity for likelihood and conventions.

Notes:
    Matching strings are declarations, not content verification or a cache key.
```

</details>

---

<a id="api-dl_complexity-results-quantity"></a>

### `Quantity`

[Исходник](../src/dl_complexity/results.py#L34)

```text
Quantity(value: 'torch.Tensor', unit: 'str', normalization: 'str' = 'total')
```

Keep a tensor together with units and normalization.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    value (torch.Tensor): Floating scalar or diagnostic tensor.
    unit (str): For example nats, bits, native_ic or probability.
    normalization (str): Total by default; per_observation is explicit.

Notes:
    The record is shallowly frozen. Tensor contents/autograd are not detached.
```

</details>

---

<a id="api-dl_complexity-results-quantity-to"></a>

#### `Quantity.to`

[Исходник](../src/dl_complexity/results.py#L64)

```text
Quantity.to(unit: 'str') -> 'Quantity'
```

Convert logarithm base without changing the mathematical estimand.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    unit (str): Same unit, or bits/nats when the source is nats/bits.

Returns:
    Quantity: New scaled tensor; same-unit conversion returns self.

Raises:
    InputError: Requested conversion would reinterpret IC, risk or a norm.

Notes:
    bits = nats / ln(2). Scaling takes O(value.numel()) work and preserves
    autograd. This is not an automatic IC-to-codelength conversion.
```

</details>

---

<a id="api-dl_complexity-results-quantity-per_observation"></a>

#### `Quantity.per_observation`

[Исходник](../src/dl_complexity/results.py#L88)

```text
Quantity.per_observation(n: 'int') -> 'Quantity'
```

Divide a total log/code/IC quantity by a declared observation count.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    n (int): Positive number of observations, not minibatches.

Returns:
    Quantity: Value/n labelled per_observation.

Raises:
    InputError: Already normalized quantity, incompatible unit or invalid n.

Notes:
    O(value.numel()) work/storage; probability bounds are not divided by n.
```

</details>

---

<a id="api-dl_complexity-results-requirementreport"></a>

### `RequirementReport`

[Исходник](../src/dl_complexity/results.py#L176)

```text
RequirementReport(
    status: 'RequirementStatus',
    missing: 'tuple[str, ...]' = (),
    actions: 'tuple[str, ...]' = (),
    assumptions: 'tuple[str, ...]' = ()
)
```

Describe preflight readiness separately from scientific assumptions.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    status (RequirementStatus): READY, PREPARATION_REQUIRED, MISSING_INPUTS or UNSUPPORTED.
    missing (tuple[str, ...]): Required input descriptions.
    actions (tuple[str, ...]): Explicit preparation steps for the caller.
    assumptions (tuple[str, ...]): Conditions not proved by readiness.
```

</details>

---

<a id="api-dl_complexity-results-requirementreport-ready"></a>

#### `RequirementReport.ready`

[Исходник](../src/dl_complexity/results.py#L192)

Свойство, читается без вызова.

Return whether status equals READY; no numerical computation.

<a id="api-dl_complexity-results-requirementstatus"></a>

### `RequirementStatus`

[Исходник](../src/dl_complexity/results.py#L167)

Name computability states without asserting theorem applicability.

- `RequirementStatus.READY` → `"READY"`
- `RequirementStatus.PREPARATION_REQUIRED` → `"PREPARATION_REQUIRED"`
- `RequirementStatus.MISSING_INPUTS` → `"MISSING_INPUTS"`
- `RequirementStatus.UNSUPPORTED` → `"UNSUPPORTED"`

<a id="api-dl_complexity-context-trainingrecipe"></a>

### `TrainingRecipe`

[Исходник](../src/dl_complexity/context.py#L15)

```text
TrainingRecipe(
    train: 'Callable',
    objective: 'str',
    config: 'Mapping[str, Any]' = <factory>
)
```

Declare a training callable and configuration without executing it.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    train (Callable): Called as train(model, **config) by run_training.
    objective (str): Human-readable declaration, not inferred from Python code.
    config (Mapping): Optimizer/data/other arguments owned by the callable.

Notes:
    A frozen record is shallow: it neither freezes tensors nor serializes closures.
```

</details>

---

<a id="api-dl_complexity-context-trainingrecord"></a>

### `TrainingRecord`

[Исходник](../src/dl_complexity/context.py#L33)

```text
TrainingRecord(
    objective: 'str',
    config: 'Mapping[str, Any]',
    rng_seed: 'int',
    status: 'str',
    source_state_id: 'str | None' = None,
    likelihood_id: 'str | None' = None
)
```

Store executed-run metadata without retaining a trained model or loss graph.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    objective (str): Declared objective.
    config (Mapping): Shallow recipe configuration snapshot.
    rng_seed (int): Outer PyTorch RNG seed, distinct from a callable's seed arg.
    status (str): Completed or failed for recorded ordinary exceptions.
    source_state_id (str, optional): User-declared starting model identity.
    likelihood_id (str, optional): User-declared likelihood identity.
```

</details>

---


<a id="reference-criteria"></a>

## 📊 Информационные критерии

Импорт: `from dl_complexity.criteria import ...`.

[AIC](#api-dl_complexity-criteria-aic-aic) | [BIC](#api-dl_complexity-criteria-bic-bic) | [HQIC](#api-dl_complexity-criteria-hqic-hqic) | [WAIC](#api-dl_complexity-criteria-waic-waic) | [InformationCriterion](#api-dl_complexity-criteria-base-informationcriterion) | [FitStatistics](#api-dl_complexity-criteria-inputs-fitstatistics) | [PosteriorLogLikelihood](#api-dl_complexity-criteria-inputs-posteriorloglikelihood)

<a id="api-dl_complexity-criteria-aic-aic"></a>

### `AIC`

[Исходник](../src/dl_complexity/criteria/aic.py#L11)

```text
AIC(*, parameter_count='trainable')
```

Compute the native information criterion `AIC`.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    -2 * log_likelihood + 2 * k

Args:
    parameter_count (str or int): Context-route parameter selection;
        inherited from FittedCriterion. Prepared inputs use their own k.

Notes:
    Supply total natural log-likelihood, not average minibatch loss.
    The observation count is optional for the total score.
    The result uses ``native_ic``, with separate deviance and penalty.
    Constructor and prepared-input evaluation take O(1) work/storage;
    a context route additionally evaluates model likelihood.

Examples:
    >>> from dl_complexity.criteria import FitStatistics
    >>> result = AIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
    >>> result.primary.unit
    'native_ic'
```

</details>

---

`parameter_count` применяется только к маршруту `context, data`: `"trainable"` считает уникальные обучаемые параметры, `"all"` — все параметры, целое число задаёт размерность явно. Переданное `FitStatistics.k` эта настройка не заменяет.

<a id="api-dl_complexity-criteria-bic-bic"></a>

### `BIC`

[Исходник](../src/dl_complexity/criteria/bic.py#L13)

```text
BIC(*, parameter_count='trainable')
```

Compute the native information criterion `BIC`.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    -2 * log_likelihood + k * log(n)

Args:
    parameter_count (str or int): Context-route parameter selection;
        inherited from FittedCriterion. Prepared inputs use their own k.

Notes:
    Supply total natural log-likelihood, not average minibatch loss.
    Half-BIC is not returned as an exact code or exact evidence.
    The result uses ``native_ic``, with separate deviance and penalty.
    Constructor and prepared-input evaluation take O(1) work/storage;
    a context route additionally evaluates model likelihood.

Examples:
    >>> from dl_complexity.criteria import FitStatistics
    >>> result = BIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
    >>> result.primary.unit
    'native_ic'
```

</details>

---

`parameter_count` применяется только к маршруту `context, data`: `"trainable"` считает уникальные обучаемые параметры, `"all"` — все параметры, целое число задаёт размерность явно. Переданное `FitStatistics.k` эта настройка не заменяет.

<a id="api-dl_complexity-criteria-hqic-hqic"></a>

### `HQIC`

[Исходник](../src/dl_complexity/criteria/hqic.py#L14)

```text
HQIC(*, parameter_count='trainable')
```

Compute the native information criterion `HQIC`.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    -2 * log_likelihood + 2 * k * log(log(n))

Args:
    parameter_count (str or int): Context-route parameter selection;
        inherited from FittedCriterion. Prepared inputs use their own k.

Notes:
    Supply total natural log-likelihood, not average minibatch loss.
    For n=2 the native penalty is negative and is not clipped.
    The result uses ``native_ic``, with separate deviance and penalty.
    Constructor and prepared-input evaluation take O(1) work/storage;
    a context route additionally evaluates model likelihood.

Examples:
    >>> from dl_complexity.criteria import FitStatistics
    >>> result = HQIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
    >>> result.primary.unit
    'native_ic'
```

</details>

---

`parameter_count` применяется только к маршруту `context, data`: `"trainable"` считает уникальные обучаемые параметры, `"all"` — все параметры, целое число задаёт размерность явно. Переданное `FitStatistics.k` эта настройка не заменяет.

<a id="api-dl_complexity-criteria-waic-waic"></a>

### `WAIC`

[Исходник](../src/dl_complexity/criteria/waic.py#L18)

```text
WAIC()
```

Compute deviance-scale WAIC2 with sample variance (ddof=1).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    lppd = sum_i log(mean_s(exp(log_likelihood[s, i])))
    p_waic = sum_i Var_s(log_likelihood[s, i], correction=1)
    elpd_waic = lppd - p_waic
    waic = -2 * elpd_waic

Notes:
    Inputs are a [S, N] matrix with S >= 2 ordinary posterior draws. The
    sample correction and observation factorisation are explicit; a single
    deterministic checkpoint cannot supply posterior uncertainty.
    This reducer performs O(S*N) work and retains tensor autograd. It does
    not implement posterior sampling or Monte Carlo error estimation.
```

</details>

---

<a id="api-dl_complexity-criteria-base-informationcriterion"></a>

### `InformationCriterion`

[Исходник](../src/dl_complexity/criteria/base.py#L10)

```text
InformationCriterion(**config)
```

Base interface for information criteria with explicit native scale.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Notes:
    Generic inputs distinguish fitted-likelihood summaries from posterior
    likelihood matrices. This class adds no fitting state or numeric work;
    ``check`` and ``estimate`` are inherited from ComplexityEstimator.
    Do not infer sklearn ``fit``/Pipeline compatibility from this hierarchy.
```

</details>

---

<a id="api-dl_complexity-criteria-inputs-fitstatistics"></a>

### `FitStatistics`

[Исходник](../src/dl_complexity/criteria/inputs.py#L10)

```text
FitStatistics(
    log_likelihood: torch.Tensor | float,
    k: int,
    n: int | None = None,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store sufficient fitted-likelihood numbers without re-evaluating a model.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    log_likelihood (float or Tensor): Scalar sum of natural log-likelihood
        over observations. No regularizer or optimizer weight decay is added.
    k (int): Nonnegative statistical parameter dimension. It is not inferred
        as an effective dimension for a singular neural network.
    n (int, optional): Number of observations, not minibatches. May be omitted
        only for unnormalised AIC.
    provenance (Provenance, optional): User-declared state/data/likelihood IDs.

Notes:
    Construction stores references in O(1); numeric validation belongs to
    the estimator. A frozen dataclass does not make a referenced Tensor
    immutable and does not certify MLE/regularity assumptions.
```

</details>

---

<a id="api-dl_complexity-criteria-inputs-posteriorloglikelihood"></a>

### `PosteriorLogLikelihood`

[Исходник](../src/dl_complexity/criteria/inputs.py#L35)

```text
PosteriorLogLikelihood(
    log_likelihood: torch.Tensor,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store pointwise log-likelihood under ordinary posterior draws.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    log_likelihood (Tensor): Floating matrix [S, N] in natural logs. Each
        column is one observation after summing its event axes; each row
        corresponds to one ordinary posterior draw, not a tempered draw.
    provenance (Provenance, optional): User-declared artifact identities.

Notes:
    WAIC2 requires S >= 2 and N >= 1. Construction is O(1), holds the matrix
    by reference, and neither samples posterior weights nor checks their
    convergence. The matrix itself requires O(S*N) caller-owned storage.
```

</details>

---


<a id="reference-bayesian"></a>

## 🧮 Байесовские оценки

Импорт: `from dl_complexity.bayesian import ...`.

[BayesianEstimator](#api-dl_complexity-bayesian-base-bayesianestimator) | [KFACFactors](#api-dl_complexity-bayesian-inputs-kfacfactors) | [KroneckerBlock](#api-dl_complexity-bayesian-inputs-kroneckerblock) | [LaplaceTerms](#api-dl_complexity-bayesian-inputs-laplaceterms) | [TemperedLikelihood](#api-dl_complexity-bayesian-inputs-temperedlikelihood) | [VariationalTerms](#api-dl_complexity-bayesian-inputs-variationalterms) | [KFACLaplace](#api-dl_complexity-bayesian-laplace-kfaclaplace) | [VariationalCode](#api-dl_complexity-bayesian-variational-variationalcode) | [WBIC](#api-dl_complexity-bayesian-wbic-wbic) | [prepare_gaussian_linear_kfac](#api-dl_complexity-bayesian-preparation-prepare_gaussian_linear_kfac) | [prepare_variational_terms](#api-dl_complexity-bayesian-preparation-prepare_variational_terms)

<a id="api-dl_complexity-bayesian-base-bayesianestimator"></a>

### `BayesianEstimator`

[Исходник](../src/dl_complexity/bayesian/base.py#L10)

```text
BayesianEstimator(**config)
```

Base interface for variational free energies and evidence approximations.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Notes:
    Family membership does not identify a single estimand or make the
    result a decodable bitstream. Typed inputs preserve distinctions among
    ordinary/tempered expectations, variational KL and posterior curvature.
    This empty semantic base adds no numeric work to ComplexityEstimator.
```

</details>

---

<a id="api-dl_complexity-bayesian-inputs-kfacfactors"></a>

### `KFACFactors`

[Исходник](../src/dl_complexity/bayesian/inputs.py#L104)

```text
KFACFactors(
    log_likelihood: torch.Tensor | float,
    log_prior: torch.Tensor | float,
    blocks: tuple[dl_complexity.bayesian.inputs.KroneckerBlock, ...],
    prior_precision: float,
    curvature_kind: str,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store Kronecker likelihood factors and exact isotropic prior addition.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    log_likelihood (float or Tensor): Scalar total natural log-likelihood.
    log_prior (float or Tensor): Scalar normalised prior log-density.
    blocks (tuple of KroneckerBlock): Nonempty selected block decomposition.
    prior_precision (float): Positive isotropic precision alpha.
    curvature_kind (str): Explicit Hessian/GGN/Fisher convention label.
    provenance (Provenance, optional): User-declared artifact identities.

Formula::

    block_eigenvalues[j, i] = scale * eig(G)[j] * eig(A)[i] + alpha

Notes:
    alpha*I is added to each implicit block exactly. This is NOT
    (A+aI) kron (G+bI), which introduces different cross terms. Construction
    stores references in O(1); no general-network factor collector is run.
```

</details>

---

<a id="api-dl_complexity-bayesian-inputs-kroneckerblock"></a>

### `KroneckerBlock`

[Исходник](../src/dl_complexity/bayesian/inputs.py#L83)

```text
KroneckerBlock(
    activation_factor: torch.Tensor,
    output_factor: torch.Tensor,
    scale: float = 1.0
)
```

Describe a likelihood precision block as scale * (G kron A).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    activation_factor (Tensor): Symmetric PSD floating matrix A, [a, a].
    output_factor (Tensor): Symmetric PSD floating matrix G, [g, g].
    scale (float): Positive multiplier converting the supplied factor
        convention to total-data likelihood precision.

Notes:
    Construction is O(1), storing references to O(a^2+g^2) input values.
    The implicit block dimension is a*g; the dense block is not formed.
    Prior addition and numerical factor validation occur in KFACLaplace.
```

</details>

---

<a id="api-dl_complexity-bayesian-inputs-laplaceterms"></a>

### `LaplaceTerms`

[Исходник](../src/dl_complexity/bayesian/inputs.py#L58)

```text
LaplaceTerms(
    log_likelihood: torch.Tensor | float,
    log_prior: torch.Tensor | float,
    dimension: int,
    logdet_precision: torch.Tensor | float,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store reference-point densities and total posterior precision logdet.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    log_likelihood (float or Tensor): Scalar total natural log-likelihood.
    log_prior (float or Tensor): Scalar normalised prior log-density at the
        same reference point, including its normalising constant.
    dimension (int): Positive selected parameter dimension d.
    logdet_precision (float or Tensor): Scalar log determinant of the total
        posterior precision, not an unscaled average-data curvature.
    provenance (Provenance, optional): User-declared artifact identities.

Notes:
    This O(1) record can bypass a new curvature pass. MAP, valid precision,
    selected subset and agreement of all terms remain caller assumptions.
```

</details>

---

<a id="api-dl_complexity-bayesian-inputs-temperedlikelihood"></a>

### `TemperedLikelihood`

[Исходник](../src/dl_complexity/bayesian/inputs.py#L10)

```text
TemperedLikelihood(
    log_likelihood: torch.Tensor,
    n: int,
    beta: float,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store total likelihood evaluations from a declared tempered posterior.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    log_likelihood (Tensor): Floating vector [S] of total natural
        log-likelihood values, one per tempered-posterior draw.
    n (int): Number of observations, greater than one.
    beta (float): Inverse temperature; WBIC requires 1/log(n).
    provenance (Provenance, optional): Declared artifact identities.

Notes:
    The target is proportional to prior * likelihood**beta. Ordinary
    posterior draws cannot silently replace this target. Construction is
    O(1) reference storage; the caller-owned vector uses O(S) memory.
    The record does not run a sampler or establish mixing/convergence.
```

</details>

---

<a id="api-dl_complexity-bayesian-inputs-variationalterms"></a>

### `VariationalTerms`

[Исходник](../src/dl_complexity/bayesian/inputs.py#L34)

```text
VariationalTerms(
    expected_nll: torch.Tensor | float,
    kl_divergence: torch.Tensor | float,
    n: int | None = None,
    kl_estimator: str = 'analytic',
    provenance: dl_complexity.results.Provenance | None = None
)
```

Store total negative-ELBO terms without implicit dataset rescaling.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    expected_nll (float or Tensor): Scalar E_q[-log p(D|theta)], total over D.
    kl_divergence (float or Tensor): Scalar KL(q||prior) estimate, in nats.
    n (int, optional): Observation count, recorded but not used to scale KL.
    kl_estimator (str): ``"analytic"``, ``"monte_carlo"`` or ``"supplied"``.
    provenance (Provenance, optional): User-declared artifact identities.

Notes:
    Construction stores references in O(1). Analytic KL must be nonnegative;
    a finite-sample Monte Carlo KL estimate may be negative. Differentiable
    tensors retain their graph; this record does not train q or the model.
```

</details>

---

<a id="api-dl_complexity-bayesian-laplace-kfaclaplace"></a>

### `KFACLaplace`

[Исходник](../src/dl_complexity/bayesian/laplace.py#L67)

```text
KFACLaplace()
```

Evaluate a Laplace free energy from total precision or KFAC factors.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    negative_log_evidence = -log_likelihood - log_prior
        - dimension/2 * log(2*pi) + logdet_precision/2

Notes:
    Inputs are either LaplaceTerms or KFACFactors. For Kronecker blocks,
    exact isotropic prior addition is used, rather than Ritter et al.'s
    factor-wise damping approximation. The reference point must be MAP to
    interpret the result as evidence; the reducer does not establish MAP.
    It neither collects general-network factors nor fits the model.
```

</details>

---

<a id="api-dl_complexity-bayesian-variational-variationalcode"></a>

### `VariationalCode`

[Исходник](../src/dl_complexity/bayesian/variational.py#L17)

```text
VariationalCode(**config)
```

Return total variational free energy, not a physical coded bitstream.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    negative_elbo = expected_nll + kl_divergence

Notes:
    Both terms are total quantities in nats. Dataset averaging, minibatch
    multiplication, KL annealing, weight decay and fitting are not inserted.
    Code interpretation needs additional coding assumptions. The inherited
    constructor stores only configuration, with no data/model execution.
```

</details>

---

`**config` сохраняет метаданные вызова. Эти ключи не задают prior, sampler или обучение: численные входы передаются в `estimate(inputs=...)`.

<a id="api-dl_complexity-bayesian-wbic-wbic"></a>

### `WBIC`

[Исходник](../src/dl_complexity/bayesian/wbic.py#L18)

```text
WBIC(**config)
```

Average total NLL under the explicitly declared WBIC temperature.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Formula::

    beta = 1 / log(n)
    wbic = -mean_s(total_log_likelihood[s])

Notes:
    No extra beta or deviance factor 2 multiplies the mean. Inputs must
    target prior * likelihood**beta; neither ordinary posterior draws nor
    repeated predictions from a deterministic checkpoint provide this.
    Sampling/mixing and the asymptotic evidence interpretation are separate
    assumptions, not established by this reducer.
```

</details>

---

`**config` сохраняет метаданные вызова. Эти ключи не задают prior, sampler или обучение: численные входы передаются в `estimate(inputs=...)`.

<a id="api-dl_complexity-bayesian-preparation-prepare_gaussian_linear_kfac"></a>

### `prepare_gaussian_linear_kfac`

[Исходник](../src/dl_complexity/bayesian/preparation.py#L81)

```text
prepare_gaussian_linear_kfac(
    model: torch.nn.modules.linear.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    noise_std: float,
    prior_precision: float,
    provenance: dl_complexity.results.Provenance | None = None
) -> dl_complexity.bayesian.inputs.KFACFactors
```

Prepare exact Gaussian-regression curvature for one plain Linear layer.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Linear): Exact built-in Linear type, with optional bias;
        all its weights/biases are selected. Must already be MAP if the
        subsequent Laplace value is to represent evidence.
    x (Tensor): Finite inputs [N, input_features], N >= 1.
    y (Tensor): Finite targets [N, output_features], matching model dtype/device.
    noise_std (float): Positive known observation standard deviation sigma.
    prior_precision (float): Positive alpha for independent N(0, alpha^-1)
        weights/biases; normalising constants are included.
    provenance (Provenance, optional): Declared artifact identities.

Returns:
    KFACFactors: One block A=X_aug.T@X_aug/N, G=I/sigma^2, scale=N;
    total likelihood and normalised prior at the supplied weights.

Raises:
    InputError: If model type, shape, dtype/device, finiteness or scales fail.

Notes:
    X_aug appends a column of ones for bias. With p augmented features and
    g outputs, work is O(N*p*g + N*p^2 + g^2), additional numeric storage
    O(N*p + N*g + p^2 + g^2 + p*g), plus autograd. The function directly
    performs the linear algebra, does not call an arbitrary model forward,
    fit parameters, build a general-network collector, or record telemetry.
```

</details>

---

<a id="api-dl_complexity-bayesian-preparation-prepare_variational_terms"></a>

### `prepare_variational_terms`

[Исходник](../src/dl_complexity/bayesian/preparation.py#L16)

```text
prepare_variational_terms(
    q: torch.distributions.distribution.Distribution,
    prior: torch.distributions.distribution.Distribution,
    total_nll: collections.abc.Callable[[torch.Tensor], torch.Tensor],
    *,
    n_samples: int = 1,
    n: int | None = None,
    analytic_kl: bool = True,
    provenance: dl_complexity.results.Provenance | None = None
) -> dl_complexity.bayesian.inputs.VariationalTerms
```

Explicitly sample a supplied q and prepare total negative-ELBO terms.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    q (Distribution): Already specified variational distribution. Uses
        rsample when available, otherwise sample without a score-function
        gradient estimator.
    prior (Distribution): Normalised prior on the same joint parameter law.
    total_nll (Callable): Function theta -> scalar total-data negative
        log-likelihood, not mean minibatch loss or regularised objective.
    n_samples (int): Positive number S of parameter draws.
    n (int, optional): Observation count recorded without implicit scaling.
    analytic_kl (bool): Request registered torch.distributions KL; otherwise
        estimate joint log q(theta) - log prior(theta) using the same draws.
    provenance (Provenance, optional): Declared artifact identities.

Returns:
    VariationalTerms: Monte Carlo expected NLL and analytic/MC KL in nats.

Raises:
    InputError: If counts/NLL scalars are invalid or requested analytic KL
        is not registered. Distribution/support errors can also propagate.

Notes:
    No fitting is performed. This explicit preparation consumes RNG. Each
    log_prob is summed to its OWN joint scalar before subtraction, avoiding
    broadcasting errors between batch-Normal and Independent representations.
    Work is sampling plus S total_nll calls and KL/density evaluation;
    storage includes all S draws and their retained autograd graphs.
```

</details>

---


<a id="reference-coding"></a>

## 📦 Кодовые длины

Импорт: `from dl_complexity.coding import ...`.

Последовательные методы, `TemperatureScaler` и `prepare_blockwise`, `prepare_online`, `prepare_replay` также доступны из `dl_complexity.coding.prequential`.

[CodeLengthEstimator](#api-dl_complexity-coding-base-codelengthestimator) | [TwoPartInputs](#api-dl_complexity-coding-inputs-twopartinputs) | [TwoPartMDL](#api-dl_complexity-coding-two_part-twopartmdl) | [prepare_two_part](#api-dl_complexity-coding-two_part-prepare_two_part) | [PrequentialEntry](#api-dl_complexity-coding-inputs-prequentialentry) | [PrequentialLedger](#api-dl_complexity-coding-inputs-prequentialledger) | [BlockwisePrequential](#api-dl_complexity-coding-prequential-blockwise-blockwiseprequential) | [OnlinePrequential](#api-dl_complexity-coding-prequential-online-onlineprequential) | [ReplayPrequential](#api-dl_complexity-coding-prequential-replay-replayprequential) | [TemperatureScaler](#api-dl_complexity-coding-prequential-calibration-temperaturescaler) | [prepare_blockwise](#api-dl_complexity-coding-prequential-protocol-prepare_blockwise) | [prepare_online](#api-dl_complexity-coding-prequential-protocol-prepare_online) | [prepare_replay](#api-dl_complexity-coding-prequential-protocol-prepare_replay)

<a id="api-dl_complexity-coding-base-codelengthestimator"></a>

### `CodeLengthEstimator`

[Исходник](../src/dl_complexity/coding/base.py#L10)

```text
CodeLengthEstimator(**config)
```

Base for explicitly defined model/data codes and predictive protocols.

<a id="api-dl_complexity-coding-inputs-twopartinputs"></a>

### `TwoPartInputs`

[Исходник](../src/dl_complexity/coding/inputs.py#L10)

```text
TwoPartInputs(
    model_bits: torch.Tensor | float,
    data_nll: torch.Tensor | float,
    n: int,
    provenance: dl_complexity.results.Provenance | None = None,
    data_code_convention: str = 'discrete',
    resolution_adjustment_nats: torch.Tensor | float = 0.0
)
```

Model bits plus data nats for the *decoded*, not original, predictor.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``discrete`` requires a nonnegative data code. For a continuous likelihood,
provide the resolution adjustment explicitly and select ``quantized``.
The adjustment is the negative total log-cell-volume, in nats. Density times
cell volume is an approximation, not an exact integration of bin probability.

Args:
    model_bits (Tensor or float): Complete nonnegative binary model cost.
    data_nll (Tensor or float): Total natural-log loss of the decoded predictor.
    n (int): Number of observations, not batches.
    provenance (Provenance, optional): User-declared state/data identities.
    data_code_convention (str): ``discrete`` or ``quantized`` density approximation.
    resolution_adjustment_nats (Tensor or float): Negative total log-cell-volume.

Notes:
    Construction stores references in O(1); it does not encode or evaluate a model.
    Frozen records do not make the referenced tensors immutable.
```

</details>

---

<a id="api-dl_complexity-coding-two_part-twopartmdl"></a>

### `TwoPartMDL`

[Исходник](../src/dl_complexity/coding/two_part.py#L19)

```text
TwoPartMDL(*, unit: str = 'nats')
```

Compute a two-part model-plus-data description length.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    unit (str): Output logarithmic unit, ``nats`` or ``bits``.

Notes:
    The inherited ``estimate`` consumes TwoPartInputs; preparation is explicitly
    separate. Scalar arithmetic costs O(1), whereas encoding and evaluating the
    decoded model generally dominate an end-to-end call.

Examples:
    >>> inputs = TwoPartInputs(model_bits=8.0, data_nll=2.0, n=2)
    >>> TwoPartMDL(unit="bits").estimate(inputs=inputs).primary.unit
    'bits'
```

</details>

---

<a id="api-dl_complexity-coding-two_part-prepare_two_part"></a>

### `prepare_two_part`

[Исходник](../src/dl_complexity/coding/two_part.py#L156)

```text
prepare_two_part(
    context,
    *,
    data,
    codec,
    model_factory=None,
    data_code_convention='discrete',
    resolution_adjustment_nats=0.0
)
```

Encode/decode explicitly, then evaluate the decoded model's likelihood.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
This is preparation, not a hidden operation of ``estimate``. Codec metadata
is counted through actual payload bytes. Codec decoding is nondifferentiable.

Args:
    context (ModelContext): Original model and explicit likelihood.
    data (DataSource): Observation batches evaluated after decoding.
    codec (ModelCodec): Encoder/decoder that charges all side information.
    model_factory (callable, optional): Optional schema-consistent factory.
    data_code_convention (str): Discrete code or quantized-density approximation.
    resolution_adjustment_nats (float or Tensor): Declared density correction.

Returns:
    TwoPartInputs: Actual payload bits and decoded-predictor data NLL.

Raises:
    InputError: If the likelihood or data conventions are invalid.
    ValueError: If the codec cannot represent or decode the model.

Notes:
    Cost is codec encode + decode + a full likelihood evaluation. It retains a
    payload, decoded model and integration copies: O(message bytes + P + buffers)
    plus forward/autograd storage. Device is relocated, but mixed dtypes are
    preserved; derivatives do not cross discrete encode/decode.
```

</details>

---

<a id="api-dl_complexity-coding-inputs-prequentialentry"></a>

### `PrequentialEntry`

[Исходник](../src/dl_complexity/coding/inputs.py#L40)

```text
PrequentialEntry(
    start: int,
    stop: int,
    nll: torch.Tensor | float,
    trained_until: int,
    calibrated_until: int = 0,
    replay_indices: tuple[int, ...] = ()
)
```

One scored interval [start, stop), before its labels enter training.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``trained_until`` and ``calibrated_until`` are exclusive prefix endpoints.
``replay_indices`` describes the update *after* scoring this interval, so
indices in the just-revealed interval are allowed, future indices are not.

Args:
    start (int): Inclusive first observation scored by this predictor.
    stop (int): Exclusive last observation, with start < stop.
    nll (Tensor or float): Total categorical code cost in nats for this interval.
    trained_until (int): Exclusive endpoint of labels used before prediction.
    calibrated_until (int): Exclusive calibration prefix endpoint.
    replay_indices (tuple[int]): Indices used only after this block was scored.

Notes:
    O(1) record creation; replay index storage is O(R) when the caller supplies R
    indices. The record itself does not prove that an external callback was causal.
```

</details>

---

<a id="api-dl_complexity-coding-inputs-prequentialledger"></a>

### `PrequentialLedger`

[Исходник](../src/dl_complexity/coding/inputs.py#L69)

```text
PrequentialLedger(
    initial_nll: torch.Tensor | float,
    initial_count: int,
    entries: tuple[dl_complexity.coding.inputs.PrequentialEntry, ...],
    n: int,
    protocol: str,
    provenance: dl_complexity.results.Provenance | None = None
)
```

Initial code and exactly-once scores for every remaining observation.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    initial_nll (Tensor or float): Initial prefix cost in nats.
    initial_count (int): Number of observations encoded by that initial cost.
    entries (tuple[PrequentialEntry]): Ordered, gap-free prediction intervals.
    n (int): Total sequence length.
    protocol (str): ``blockwise``, ``online`` or ``replay``.
    provenance (Provenance, optional): User-declared identities.

Notes:
    O(1) wrapper creation; retained storage is O(B + R_total) for B intervals
    and all recorded replay indices, in addition to the caller-owned sequence.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-blockwise-blockwiseprequential"></a>

### `BlockwisePrequential`

[Исходник](../src/dl_complexity/coding/prequential/blockwise.py#L6)

```text
BlockwisePrequential(*, unit='nats')
```

Evaluate a blockwise ledger for independently retrained prefix predictors.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    unit (str): Inherited output unit, ``nats`` or ``bits``.

Notes:
    ``prepare_blockwise`` constructs the ledger through explicit callbacks.
    The inherited estimate validates B intervals and R recorded replay indices,
    then sums B scalar costs in O(B + R) time and O(B) temporary storage.
    It never reconstructs prefix predictions from final checkpoint weights.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-online-onlineprequential"></a>

### `OnlinePrequential`

[Исходник](../src/dl_complexity/coding/prequential/online.py#L6)

```text
OnlinePrequential(*, unit='nats')
```

Evaluate a online ledger for one incrementally updated predictor.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    unit (str): Inherited output unit, ``nats`` or ``bits``.

Notes:
    ``prepare_online`` constructs the ledger through explicit callbacks.
    The inherited estimate validates B intervals and R recorded replay indices,
    then sums B scalar costs in O(B + R) time and O(B) temporary storage.
    It never reconstructs prefix predictions from final checkpoint weights.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-replay-replayprequential"></a>

### `ReplayPrequential`

[Исходник](../src/dl_complexity/coding/prequential/replay.py#L6)

```text
ReplayPrequential(*, unit='nats')
```

Evaluate a replay ledger for one predictor updated with revealed-prefix replay.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    unit (str): Inherited output unit, ``nats`` or ``bits``.

Notes:
    ``prepare_replay`` constructs the ledger through explicit callbacks.
    The inherited estimate validates B intervals and R recorded replay indices,
    then sums B scalar costs in O(B + R) time and O(B) temporary storage.
    It never reconstructs prefix predictions from final checkpoint weights.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-calibration-temperaturescaler"></a>

### `TemperatureScaler`

[Исходник](../src/dl_complexity/coding/prequential/calibration.py#L16)

```text
TemperatureScaler(
    temperature: float | torch.Tensor = 1.0,
    *,
    steps: int = 30,
    learning_rate: float = 0.05
)
```

Fit a positive scalar temperature on already-revealed categorical labels.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    temperature (float or Tensor): Positive initial scalar; copied and detached.
    steps (int): Adam steps per explicit update.
    learning_rate (float): Positive finite optimizer learning rate.

Notes:
    log_probs uses log_softmax(logits / T). update optimizes log(T) with fixed
    logits, so it cannot train the network. No observations are retained.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-calibration-temperaturescaler-log_probs"></a>

#### `TemperatureScaler.log_probs`

[Исходник](../src/dl_complexity/coding/prequential/calibration.py#L56)

```text
TemperatureScaler.log_probs(logits: torch.Tensor) -> torch.Tensor
```

Compute normalized log probabilities using the current temperature.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    logits (Tensor): Finite floating logits of shape [batch, classes >= 2].

Returns:
    Tensor: Natural-log probabilities with unchanged shape, dtype and device.

Raises:
    InputError: If logits are nonfinite or incorrectly shaped.

Notes:
    O(batch * classes) time and storage. Gradients may flow into supplied logits;
    the stored temperature is detached from any previous calibration graph.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-calibration-temperaturescaler-update"></a>

#### `TemperatureScaler.update`

[Исходник](../src/dl_complexity/coding/prequential/calibration.py#L81)

```text
TemperatureScaler.update(logits: torch.Tensor, targets: torch.Tensor)
```

Fit temperature on this already-scored batch; store no event history.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    logits (Tensor): Already-scored finite logits [batch, classes], detached here.
    targets (LongTensor): Revealed labels [batch] on the logits' device.

Returns:
    TemperatureScaler: This instance, with a newly fitted detached temperature.

Raises:
    InputError: If inputs are invalid or temperature fitting diverges.

Notes:
    For A optimizer steps the work is O(A * batch * classes), with
    O(batch * classes) intermediates and O(1) optimizer parameter state.
    ``fit`` is an alias for this method, not a second fitting implementation.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-calibration-temperaturescaler-fit"></a>

#### `TemperatureScaler.fit`

[Исходник](../src/dl_complexity/coding/prequential/calibration.py#L81)

```text
TemperatureScaler.fit(logits: torch.Tensor, targets: torch.Tensor)
```

Псевдоним `TemperatureScaler.update` с тем же контрактом.

<a id="api-dl_complexity-coding-prequential-protocol-prepare_blockwise"></a>

### `prepare_blockwise`

[Исходник](../src/dl_complexity/coding/prequential/protocol.py#L171)

```text
prepare_blockwise(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    boundaries: collections.abc.Sequence[int],
    model_factory: collections.abc.Callable,
    train: collections.abc.Callable,
    n_classes: int,
    predict_logits: collections.abc.Callable | None = None,
    initial_code=None,
    calibrator_factory: collections.abc.Callable | None = None,
    provenance=None
)
```

Chunk-incremental/from-scratch code, with prefix-only training.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``boundaries`` contains exclusive block ends, ending at n. The first block
uses the declared initial code (uniform alphabet by default). Each later
block uses a fresh ``model_factory()`` trained on exactly ``x[:start]`` and
``y[:start]``. ``train`` returns a replacement state or None for in-place use.

Args:
    x (Tensor): Ordered observations [n, ...].
    y (LongTensor): Ordered labels [n].
    boundaries (Sequence[int]): Strictly increasing exclusive block ends, last n.
    model_factory (callable): Fresh learner state for every noninitial block.
    train (callable): (state, prefix_x, prefix_y) -> replacement state or None.
    n_classes (int): Fixed shared alphabet size.
    predict_logits (callable, optional): (state, x) -> logits [batch, classes].
    initial_code (float or Tensor, optional): Prefix cost; uniform by default.
    calibrator_factory (callable, optional): Fresh past-only calibrator per block.
    provenance (Provenance, optional): User-declared identities.

Returns:
    PrequentialLedger: Detached categorical costs and causal interval metadata.

Raises:
    InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

Notes:
    Callbacks receive only the prefix/revealed slices, but their closed-over
    state cannot be audited. Input tensors are already materialized; this is
    not an out-of-core streaming data loader. Costs of user training callbacks
    must be added explicitly to any arithmetic complexity estimate.
    For block starts s_j and sizes b_j, work is sum_j Train(s_j)+F(b_j),
    plus optional calibration on each prefix. Persistent ledger storage is O(B);
    weak-reference bookkeeping does not retain all historical models.
```

</details>

---

<a id="api-dl_complexity-coding-prequential-protocol-prepare_online"></a>

### `prepare_online`

[Исходник](../src/dl_complexity/coding/prequential/protocol.py#L271)

```text
prepare_online(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model_factory: collections.abc.Callable,
    update: collections.abc.Callable,
    n_classes: int,
    batch_size: int = 1,
    predict_logits: collections.abc.Callable | None = None,
    initial_count: int = 0,
    initial_code=None,
    calibrator=None,
    provenance=None
)
```

Online next-minibatch code: score, calibrator update, learner update.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
The callback is ``update(state, revealed_x, revealed_y)``. Only the new
scored minibatch is passed. Persistent learner/optimizer state is owned by
the caller's state object; this function does not infer an optimizer.

Args:
    x (Tensor): Materialized observations [n, ...].
    y (LongTensor): Labels [n].
    model_factory (callable): Creates the initial learner state.
    update (callable): (state, new_x, new_y) -> state or None, after scoring.
    n_classes (int): Fixed shared alphabet size.
    batch_size (int): Positive next-block size.
    predict_logits (callable, optional): Explicit state-to-logits adapter.
    initial_count (int): Initially encoded prefix length, possibly zero.
    initial_code (float or Tensor, optional): Prefix code in nats.
    calibrator (object, optional): Mutable past-only temperature component.
    provenance (Provenance, optional): User-declared identities.

Returns:
    PrequentialLedger: Detached categorical costs and causal interval metadata.

Raises:
    InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

Notes:
    Callbacks receive only the prefix/revealed slices, but their closed-over
    state cannot be audited. Input tensors are already materialized; this is
    not an out-of-core streaming data loader. Costs of user training callbacks
    must be added explicitly to any arithmetic complexity estimate.
    Work is sum_j[F(b_j)+Update(b_j)+Calibration(b_j)], plus O(n*C)
    categorical reductions. Ledger storage is O(ceil(n/batch_size)).
```

</details>

---

<a id="api-dl_complexity-coding-prequential-protocol-prepare_replay"></a>

### `prepare_replay`

[Исходник](../src/dl_complexity/coding/prequential/protocol.py#L334)

```text
prepare_replay(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model_factory: collections.abc.Callable,
    update: collections.abc.Callable,
    n_classes: int,
    replay_size: int,
    batch_size: int = 1,
    predict_logits: collections.abc.Callable | None = None,
    initial_count: int = 0,
    initial_code=None,
    calibrator=None,
    replay_policy: collections.abc.Callable | None = None,
    seed: int = 0,
    provenance=None
)
```

Uniform-prefix replay code; replay losses are not encoded again.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Callback: ``update(state, new_x, new_y, replay_x, replay_y)``. Replay is drawn
after scoring the new minibatch, from indices below its exclusive stop.
A custom ``replay_policy(revealed_count)`` may return only such indices.
This is exact revealed-prefix random replay, not the paper's replay-streams
memory optimization or a claim to reproduce its empirical experiments.

Args:
    x (Tensor): Materialized observations [n, ...].
    y (LongTensor): Labels [n].
    model_factory (callable): Creates callback-owned learner state.
    update (callable): (state, new_x, new_y, replay_x, replay_y) -> state or None.
    n_classes (int): Declared alphabet size.
    replay_size (int): Maximum sampled revealed indices per update.
    batch_size (int): Positive number of newly scored observations.
    predict_logits (callable, optional): Explicit prediction adapter.
    initial_count (int): Uniform/explicit-code prefix length.
    initial_code (float or Tensor, optional): Initial prefix cost in nats.
    calibrator (object, optional): Updated only after scoring a batch.
    replay_policy (callable, optional): (revealed_count) -> allowed indices.
    seed (int): Seed for the local replay generator.
    provenance (Provenance, optional): User-declared identities.

Returns:
    PrequentialLedger: Detached categorical costs and causal interval metadata.

Raises:
    InputError: If sequence, logits, boundaries or callback-produced metadata are invalid.

Notes:
    Callbacks receive only the prefix/revealed slices, but their closed-over
    state cannot be audited. Input tensors are already materialized; this is
    not an out-of-core streaming data loader. Costs of user training callbacks
    must be added explicitly to any arithmetic complexity estimate.
    Default randperm(stop) costs O(stop) per update, even for a tiny replay_size:
    O(sum_j stop_j), up to O(n^2/batch_size). Retained replay indices cost
    O(B*min(replay_size,n)); selected data batches allocate gather copies.
```

</details>

---


<a id="reference-compression"></a>

## 🗜️ Преобразования и кодеки

Импорт: `from dl_complexity.compression import ...`.

[EncodedModel](#api-dl_complexity-compression-base-encodedmodel) | [ModelCodec](#api-dl_complexity-compression-base-modelcodec) | [ModelTransform](#api-dl_complexity-compression-base-modeltransform) | [BinaryModelCodec](#api-dl_complexity-compression-codecs-binarymodelcodec) | [LowRankApproximation](#api-dl_complexity-compression-low_rank-lowrankapproximation) | [MagnitudePruning](#api-dl_complexity-compression-pruning-magnitudepruning) | [QuantizedTensor](#api-dl_complexity-compression-quantization-quantizedtensor) | [UniformQuantization](#api-dl_complexity-compression-quantization-uniformquantization) | [uniform_quantize](#api-dl_complexity-compression-quantization-uniform_quantize) | [AroraMatrixProject](#api-dl_complexity-compression-arora-aroramatrixproject) | [MatrixProjection](#api-dl_complexity-compression-arora-matrixprojection) | [ProjectedModel](#api-dl_complexity-compression-arora-projectedmodel) | [make_projection_helpers](#api-dl_complexity-compression-arora-make_projection_helpers) | [matrix_project](#api-dl_complexity-compression-arora-matrix_project) | [nominal_projection_count](#api-dl_complexity-compression-arora-nominal_projection_count)

<a id="api-dl_complexity-compression-base-encodedmodel"></a>

### `EncodedModel`

[Исходник](../src/dl_complexity/compression/base.py#L10)

```text
EncodedModel(payload: bytes)
```

An actual binary message, including its decoder/architecture metadata.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    payload (bytes): Immutable, complete versioned message.

Notes:
    Wrapping a bytes object is O(1); the message owns O(message bytes) storage.
```

</details>

---

<a id="api-dl_complexity-compression-base-encodedmodel-length_bits"></a>

#### `EncodedModel.length_bits`

[Исходник](../src/dl_complexity/compression/base.py#L32)

Свойство, читается без вызова.

Physical bit count, including byte padding and all headers.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    int: Exactly 8 * len(payload), computed in O(1) without parsing the message.
```

</details>

---

<a id="api-dl_complexity-compression-base-modelcodec"></a>

### `ModelCodec`

[Исходник](../src/dl_complexity/compression/base.py#L58)

```text
ModelCodec()
```

Encode/decode a predictor under an explicitly specified coding language.

<a id="api-dl_complexity-compression-base-modelcodec-encode"></a>

#### `ModelCodec.encode`

[Исходник](../src/dl_complexity/compression/base.py#L61)

```text
ModelCodec.encode(model: torch.nn.modules.module.Module) -> dl_complexity.compression.base.EncodedModel
```

Return a complete binary description, not a `torch.save` proxy.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Predictor whose permitted structure and state are encoded.

Returns:
    EncodedModel: Complete model message; code length includes metadata.
```

</details>

---

<a id="api-dl_complexity-compression-base-modelcodec-decode"></a>

#### `ModelCodec.decode`

[Исходник](../src/dl_complexity/compression/base.py#L72)

```text
ModelCodec.decode(
    encoded: dl_complexity.compression.base.EncodedModel,
    model_factory: Optional[Callable[[], torch.nn.modules.module.Module]] = None
) -> torch.nn.modules.module.Module
```

Reconstruct a predictor using only the message and shared codec rules.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    encoded (EncodedModel): Message from a declared coding language.
    model_factory (callable, optional): Convenience factory, not free side information.

Returns:
    nn.Module: Predictor reconstructed from charged bytes and shared codec rules.
```

</details>

---

<a id="api-dl_complexity-compression-base-modeltransform"></a>

### `ModelTransform`

[Исходник](../src/dl_complexity/compression/base.py#L42)

```text
ModelTransform()
```

A declared, non-mutating transformation of a PyTorch model.

<a id="api-dl_complexity-compression-base-modeltransform-apply"></a>

#### `ModelTransform.apply`

[Исходник](../src/dl_complexity/compression/base.py#L45)

```text
ModelTransform.apply(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Return an independent transformed model, leaving `model` unchanged.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Source predictor, not modified by the transform.

Returns:
    nn.Module: Independent transformed model. Concrete transforms document
    their numerical approximation, device support and O(P) copy cost.
```

</details>

---

<a id="api-dl_complexity-compression-codecs-binarymodelcodec"></a>

### `BinaryModelCodec`

[Исходник](../src/dl_complexity/compression/codecs.py#L447)

```text
BinaryModelCodec(
    encoding: str = 'dense',
    *,
    transform: dl_complexity.compression.base.ModelTransform | None = None,
    bits: int = 8,
    rank: int = 1
)
```

Encode supported built-in PyTorch models in a versioned binary language.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``encoding`` selects dense values, sparse masks+values, matrix SVD factors,
or packed affine quantization. Architecture, names, dtypes, shapes, frozen
flags, padding, factors/scales and header lengths are all charged. Shared
parameters/storage and custom architectures are rejected, not misdescribed.
Dense/sparse code is lossless numerically; sparse code canonicalizes -0.
Low-rank and quantized codes are explicitly lossy. The format is primarily
a correctness reference, not a competitive production compressor.

Args:
    encoding (str): ``dense``, ``sparse``, ``low_rank`` or ``quantized``.
    transform (ModelTransform, optional): Explicit non-mutating transform before coding.
    bits (int): Quantized integer width between one and sixteen.
    rank (int): Maximum matrix factor rank in low-rank coding.

Notes:
    The format is a code language, not a claim of entropy-optimal compression.
    The default decoder restores CPU tensors; caller-visible relocation is
    explicit. Fixed decoder implementation is shared; per-model schema is charged.
```

</details>

---

<a id="api-dl_complexity-compression-codecs-binarymodelcodec-encode"></a>

#### `BinaryModelCodec.encode`

[Исходник](../src/dl_complexity/compression/codecs.py#L503)

```text
BinaryModelCodec.encode(model: torch.nn.modules.module.Module) -> dl_complexity.compression.base.EncodedModel
```

Encode architecture and complete state into a physical binary message.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Supported built-in model with nonaliased state storage.

Returns:
    EncodedModel: Magic/version, 8-byte header length, JSON metadata and tensor chunks.

Raises:
    ValueError: If architecture, state, aliasing or host byte order is unsupported.

Notes:
    Includes optional transform cost and per-tensor encoding work. Metadata,
    names, shapes, modes, masks/factors/scales and padding are all charged.
    Byte chunks and final concatenation may coexist, giving O(message bytes)
    extra storage beyond any transformed model, SVD and Python-list costs.
```

</details>

---

<a id="api-dl_complexity-compression-codecs-binarymodelcodec-decode"></a>

#### `BinaryModelCodec.decode`

[Исходник](../src/dl_complexity/compression/codecs.py#L559)

```text
BinaryModelCodec.decode(
    encoded: dl_complexity.compression.base.EncodedModel,
    model_factory: collections.abc.Callable[[], torch.nn.modules.module.Module] | None = None
) -> torch.nn.modules.module.Module
```

Parse the complete message and rebuild the transmitted predictor.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    encoded (EncodedModel): Versioned physical message.
    model_factory (callable, optional): Must agree with the charged schema; it
        cannot smuggle in a custom uncharged predictor.

Returns:
    nn.Module: Fresh CPU model with decoded dtypes, state and frozen/mode flags.

Raises:
    ValueError: If headers, chunks, schema or state assignment are inconsistent.

Notes:
    Parsing/copying costs O(message bytes) besides tensor-specific unpack/SVD
    reconstruction costs. Peak storage includes message chunks and decoded
    tensors. No pickle is executed; this is not a malicious-input resource sandbox.
```

</details>

---

<a id="api-dl_complexity-compression-low_rank-lowrankapproximation"></a>

### `LowRankApproximation`

[Исходник](../src/dl_complexity/compression/low_rank.py#L11)

```text
LowRankApproximation(rank: int)
```

Truncate each two-dimensional weight matrix to at most `rank`.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Convolutional kernels and biases are not silently flattened. The returned
network has the same forward architecture; a low-rank codec stores factors
instead of dense reconstructions to obtain an actual model description.
```

</details>

---

<a id="api-dl_complexity-compression-low_rank-lowrankapproximation-apply"></a>

#### `LowRankApproximation.apply`

[Исходник](../src/dl_complexity/compression/low_rank.py#L35)

```text
LowRankApproximation.apply(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Copy the model and replace each matrix by its truncated SVD reconstruction.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Finite float32/float64 two-dimensional weight parameters.

Returns:
    nn.Module: Same architecture with approximate dense matrices; kernels/biases stay.

Raises:
    ValueError: If matrix dtype or finite-value checks are unsupported.

Notes:
    For matrix m*n and d=min(m,n), economy SVD costs O(m*n*d); storage for U/S/Vh
    is O((m+n)*d). Reconstruction adds O(m*n*rank) work and O(m*n) storage.
    The model copy costs O(P); this is not a randomized or truncated-only SVD.
    ``no_grad`` makes this preparation nondifferentiable through the original.
```

</details>

---

<a id="api-dl_complexity-compression-pruning-magnitudepruning"></a>

### `MagnitudePruning`

[Исходник](../src/dl_complexity/compression/pruning.py#L12)

```text
MagnitudePruning(fraction: float = 0.5)
```

Set the smallest `floor(fraction * N)` matrix/kernel weights to zero.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Selection is global across parameters with ndim >= 2; biases are untouched.
Ties use stable parameter/flat-index order. This transformation does not by
itself define a description length: use an explicit sparse codec afterward.
```

</details>

---

<a id="api-dl_complexity-compression-pruning-magnitudepruning-apply"></a>

#### `MagnitudePruning.apply`

[Исходник](../src/dl_complexity/compression/pruning.py#L40)

```text
MagnitudePruning.apply(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Copy the model and zero globally smallest eligible weights with stable ties.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Finite weights; eligible ndim >= 2 parameters share a device.

Returns:
    nn.Module: Independent model; biases and ineligible parameters are unchanged.

Raises:
    ValueError: If eligible weights are nonfinite or span several devices.

Notes:
    For P copied parameters and N eligible weights: O(P + N*log(N)) time,
    O(P + N) extra storage. A full stable argsort is used, not linear selection.
    Magnitudes are explicitly float64 for comparison; returned dtype/device
    are preserved. Discrete mask selection does not retain autograd.
```

</details>

---

<a id="api-dl_complexity-compression-quantization-quantizedtensor"></a>

### `QuantizedTensor`

[Исходник](../src/dl_complexity/compression/quantization.py#L13)

```text
QuantizedTensor(
    codes: torch.Tensor,
    minimum: float,
    scale: float,
    bits: int,
    dtype: torch.dtype
)
```

Integer codes plus the metadata required for deterministic decoding.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    codes (LongTensor): Integer level indices, with the original tensor shape.
    minimum (float): Transmitted lower endpoint, not free side information.
    scale (float): Positive transmitted spacing between consecutive levels.
    bits (int): Declared level-index bit width.
    dtype (torch.dtype): Floating dtype restored by decoding.

Notes:
    The wrapper is O(1); codes retain O(N) int64 storage, not a packed bitstream.
    Use BinaryModelCodec to measure physical packed length.
```

</details>

---

<a id="api-dl_complexity-compression-quantization-quantizedtensor-dequantize"></a>

#### `QuantizedTensor.dequantize`

[Исходник](../src/dl_complexity/compression/quantization.py#L35)

```text
QuantizedTensor.dequantize() -> torch.Tensor
```

Decode on the codes' device and restore the original floating dtype.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    Tensor: codes * scale + minimum, cast to the original dtype on the codes' device.

Notes:
    O(N) arithmetic/storage. Float64 intermediate arithmetic is deliberate;
    rounding/index selection in the encoder is not differentiated.
```

</details>

---

<a id="api-dl_complexity-compression-quantization-uniformquantization"></a>

### `UniformQuantization`

[Исходник](../src/dl_complexity/compression/quantization.py#L92)

```text
UniformQuantization(bits: int = 8)
```

Replace floating parameters by their decoded affine-grid values.

<a id="api-dl_complexity-compression-quantization-uniformquantization-apply"></a>

#### `UniformQuantization.apply`

[Исходник](../src/dl_complexity/compression/quantization.py#L111)

```text
UniformQuantization.apply(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Copy the model and quantize/dequantize its nonempty floating parameters.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Predictor whose parameters may use supported floating dtypes.

Returns:
    nn.Module: Independent dense model with rounded parameter values.

Notes:
    O(P) model copy and elementwise work; peak extra storage includes the copy
    plus float64/int64 intermediates for the largest tensor. Buffers are not
    transformed here. No QAT, activation calibration or fine-tuning is performed.
```

</details>

---

<a id="api-dl_complexity-compression-quantization-uniform_quantize"></a>

### `uniform_quantize`

[Исходник](../src/dl_complexity/compression/quantization.py#L48)

```text
uniform_quantize(tensor: torch.Tensor, bits: int = 8) -> dl_complexity.compression.quantization.QuantizedTensor
```

Quantize to 2**bits levels using min/max and nearest-even rounding.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
The range/scale are learned values, not free side information: a codec must
transmit them. Float64 arithmetic is explicit here for range calculation;
the decoded tensor returns to the original dtype and device.

Args:
    tensor (Tensor): Nonempty finite floating tensor with N elements.
    bits (int): Number of bits per index, between one and sixteen.

Returns:
    QuantizedTensor: Int64 indices plus min/scale/original-dtype metadata.

Raises:
    ValueError: If bit width, input values or representable range are invalid.

Notes:
    O(N) reductions and elementwise work, O(N) float64 and int64 intermediates.
    Constant tensors use scale=1 and zero codes. Scalar min/max extraction may
    synchronize an accelerator; the source tensor is detached, not mutated.
```

</details>

---

<a id="api-dl_complexity-compression-arora-aroramatrixproject"></a>

### `AroraMatrixProject`

[Исходник](../src/dl_complexity/compression/arora.py#L493)

```text
AroraMatrixProject(
    helpers: Mapping[str, torch.Tensor],
    coefficient_alphabet: torch.Tensor
)
```

Project all weights of a bias-free small Linear/ReLU MLP.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Constructor stores only fixed helper/alphabet configuration. ``project``
returns learned finite indices and an independent model; no fit artifact
is cached inside this object. Unlike data-derived min/max quantization, a
fixed alphabet gives a declared r**q family conditional on fixed helpers.
```

</details>

---

<a id="api-dl_complexity-compression-arora-aroramatrixproject-project"></a>

#### `AroraMatrixProject.project`

[Исходник](../src/dl_complexity/compression/arora.py#L516)

```text
AroraMatrixProject.project(model: torch.nn.modules.module.Module) -> dl_complexity.compression.arora.ProjectedModel
```

Project every weight matrix and bind a clean finite-decoder architecture.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Bias-free supported model matching the configured helpers.

Returns:
    ProjectedModel: Convenience model, finite indices and canonical schema.

Raises:
    ValueError: If model/helper names, state or numerical values are unsupported.

Notes:
    O(P) model copy plus sum_l O(K_l*P_l + K_l*r) projection work. The result
    retains helpers and indices; it is not automatically a smaller dense model.
```

</details>

---

<a id="api-dl_complexity-compression-arora-aroramatrixproject-apply"></a>

#### `AroraMatrixProject.apply`

[Исходник](../src/dl_complexity/compression/arora.py#L547)

```text
AroraMatrixProject.apply(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Return only the convenience projected model from the richer project result.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Supported source model.

Returns:
    nn.Module: Independent projected predictor.

Notes:
    Has the same projection cost as project. Use project instead when a later
    Arora certificate must derive q/r from actual finite indices.
```

</details>

---

<a id="api-dl_complexity-compression-arora-matrixprojection"></a>

### `MatrixProjection`

[Исходник](../src/dl_complexity/compression/arora.py#L209)

```text
MatrixProjection(
    coefficient_indices: torch.Tensor,
    coefficient_alphabet: torch.Tensor,
    helper: torch.Tensor,
    max_coefficient_rounding_error: torch.Tensor
)
```

A matrix represented by finite coefficients in a fixed helper basis.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    coefficient_indices (LongTensor): K indices into the fixed alphabet.
    coefficient_alphabet (Tensor): Strictly ordered finite grid [r].
    helper (Tensor): Fixed Rademacher matrices [K, rows, cols].
    max_coefficient_rounding_error (Tensor): Largest scalar quantization error.

Notes:
    Record creation is O(1). Referenced storage is O(K*P + r + K); freezing the
    dataclass does not freeze Tensor contents. Reconstruction is computed afresh.
```

</details>

---

<a id="api-dl_complexity-compression-arora-matrixprojection-reconstructed"></a>

#### `MatrixProjection.reconstructed`

[Исходник](../src/dl_complexity/compression/arora.py#L229)

Свойство, читается без вызова.

Decode finite indices in the fixed helper basis, without fitted state.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    Tensor: mean_k alphabet[index_k] * helper_k, with shape [rows, cols].

Notes:
    Every property access performs O(K*P) multiplication/reduction and allocates
    a broadcast [K, rows, cols] temporary; this is not an O(1) cached attribute.
```

</details>

---

<a id="api-dl_complexity-compression-arora-matrixprojection-parameter_count"></a>

#### `MatrixProjection.parameter_count`

[Исходник](../src/dl_complexity/compression/arora.py#L243)

Свойство, читается без вызова.

Return K, the number of finite coefficient coordinates, in O(1).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    int: Number of stored coefficient indices, not number of dense weights.
```

</details>

---

<a id="api-dl_complexity-compression-arora-matrixprojection-values_per_parameter"></a>

#### `MatrixProjection.values_per_parameter`

[Исходник](../src/dl_complexity/compression/arora.py#L252)

Свойство, читается без вызова.

Return fixed alphabet size r in O(1).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    int: Maximum number of values for each finite coefficient coordinate.
```

</details>

---

<a id="api-dl_complexity-compression-arora-projectedmodel"></a>

### `ProjectedModel`

[Исходник](../src/dl_complexity/compression/arora.py#L262)

```text
ProjectedModel(
    model: torch.nn.modules.module.Module,
    projections: Mapping[str, dl_complexity.compression.arora.MatrixProjection],
    architecture_json: str
)
```

Independent projected model and the actual finite representation sizes.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Independent convenience predictor matching finite indices.
    projections (Mapping[str, MatrixProjection]): Complete parameter-name mapping.
    architecture_json (str): Immutable canonical architecture binding.

Notes:
    The record owns references, not deep immutability of tensors or model. decode
    revalidates all mutable state and creates a clean predictor before certification.
```

</details>

---

<a id="api-dl_complexity-compression-arora-projectedmodel-decode"></a>

#### `ProjectedModel.decode`

[Исходник](../src/dl_complexity/compression/arora.py#L280)

```text
ProjectedModel.decode() -> torch.nn.modules.module.Module
```

Rebuild g from the bound schema and finite indices, not mutable forward.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
The convenience model is revalidated and must still match both schema
and code. No user-provided instance, hook, extra buffer or overridden
forward is used to produce the certified compressed logits.

Returns:
    nn.Module: Fresh predictor built only from canonical schema and finite indices.

Raises:
    ValueError: If schema, aliases, dtype/device or convenience weights changed.

Notes:
    O(sum_l K_l*P_l + schema work) time and similarly sized broadcast temporaries,
    plus O(P) decoded state copies. Mutable instance forwards are never trusted.
```

</details>

---

<a id="api-dl_complexity-compression-arora-projectedmodel-parameter_count"></a>

#### `ProjectedModel.parameter_count`

[Исходник](../src/dl_complexity/compression/arora.py#L320)

Свойство, читается без вызова.

Sum all finite coefficient coordinates in O(number of weight matrices).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    int: q for the declared finite predictor family.
```

</details>

---

<a id="api-dl_complexity-compression-arora-projectedmodel-values_per_parameter"></a>

#### `ProjectedModel.values_per_parameter`

[Исходник](../src/dl_complexity/compression/arora.py#L331)

Свойство, читается без вызова.

Find the largest fixed coefficient alphabet in O(number of matrices).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Returns:
    int: r such that product_l r_l**K_l <= r**q.
```

</details>

---

<a id="api-dl_complexity-compression-arora-make_projection_helpers"></a>

### `make_projection_helpers`

[Исходник](../src/dl_complexity/compression/arora.py#L413)

```text
make_projection_helpers(
    model: torch.nn.modules.module.Module,
    n_projections: int,
    *,
    seed: int
) -> dict[str, torch.Tensor]
```

Generate fixed Rademacher helpers using a local CPU generator.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Only architecture shapes/dtypes/devices are read, not learned weights.
Calling this after training does not establish independence by itself: the
user must have fixed seed, architecture and projection count before data.

Args:
    model (nn.Module): Supported bias-free model whose architecture is already fixed.
    n_projections (int): Positive K helper matrices per weight matrix.
    seed (int): Independently predeclared local CPU generator seed.

Returns:
    dict[str, Tensor]: Parameter-name -> [K, rows, cols] Rademacher helpers.

Raises:
    ValueError: If model language or configuration is unsupported.

Notes:
    O(K*P) generation/storage, including CPU integer tensors and device copies.
    It does not consume global RNG or read weight values; late seed selection
    still cannot establish independence of the helper string from the sample.
```

</details>

---

<a id="api-dl_complexity-compression-arora-matrix_project"></a>

### `matrix_project`

[Исходник](../src/dl_complexity/compression/arora.py#L343)

```text
matrix_project(
    matrix: torch.Tensor,
    helper: torch.Tensor,
    coefficient_alphabet: torch.Tensor
) -> dl_complexity.compression.arora.MatrixProjection
```

Compute sum_k <A,M_k> M_k / K, after declared finite-grid rounding.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``helper`` has shape [K, rows, cols] with fixed Rademacher entries +/-1.
``coefficient_alphabet`` is a finite, sorted, data-independent grid. Nearest
grid value is selected; exact ties select its lower index. Quantization may
increase error, which MUST be checked on all training logits afterward.
No probability certificate is fabricated from a requested projection size.

Args:
    matrix (Tensor): Finite floating matrix [rows, cols].
    helper (Tensor): Fixed +/-1 matrices [K, rows, cols], same dtype/device.
    coefficient_alphabet (Tensor): Fixed strictly increasing grid [r].

Returns:
    MatrixProjection: Finite indices, copied helper/grid and rounding diagnostic.

Raises:
    ValueError: If values, shapes, dtype/device or coefficient arithmetic are invalid.

Notes:
    Unquantized z_k = <A,M_k>; each z_k is replaced by its nearest grid point.
    Coefficients cost O(K*P), nearest-grid search O(K*r), with corresponding
    broadcast temporaries. Returned helper clones cost O(K*P); all work is no_grad.
```

</details>

---

<a id="api-dl_complexity-compression-arora-nominal_projection_count"></a>

### `nominal_projection_count`

[Исходник](../src/dl_complexity/compression/arora.py#L465)

```text
nominal_projection_count(epsilon: float, eta: float) -> int
```

Paper Algorithm 1 budget ceil(log(1/eta)/epsilon**2).

<details>
<summary>Параметры, результат и ограничения</summary>

```text
This is the algorithm's nominal budget, not an independently verified
finite-sample failure-probability guarantee for quantized coefficients.
Theorem 2.1 still needs a measured sample-compressibility certificate.

Args:
    epsilon (float): Nominal error parameter in (0, 1].
    eta (float): Nominal failure probability in (0, 1).

Returns:
    int: ceil(-log(eta) / epsilon**2).

Raises:
    ValueError: If error/failure parameters are outside their declared range.

Notes:
    O(1) arithmetic. This is not a validated allocation of a complete network's
    geometry-dependent Theorem 4.1 budget or a guarantee after grid rounding.
```

</details>

---


<a id="reference-bounds"></a>

## 📐 Границы риска

Импорт: `from dl_complexity.bounds import ...`.

[GeneralizationBoundEstimator](#api-dl_complexity-bounds-base-generalizationboundestimator) | [AroraCompressionBound](#api-dl_complexity-bounds-compression-aroracompressionbound) | [AroraCompressionInputs](#api-dl_complexity-bounds-inputs-aroracompressioninputs) | [arora_inputs_from_logits](#api-dl_complexity-bounds-compression-arora_inputs_from_logits) | [prepare_arora_compression](#api-dl_complexity-bounds-compression-prepare_arora_compression) | [prepare_arora_projection](#api-dl_complexity-bounds-compression-prepare_arora_projection)

<a id="api-dl_complexity-bounds-base-generalizationboundestimator"></a>

### `GeneralizationBoundEstimator`

[Исходник](../src/dl_complexity/bounds/base.py#L10)

```text
GeneralizationBoundEstimator(**config)
```

A bound must identify its theorem, predictor and confidence convention.

<a id="api-dl_complexity-bounds-compression-aroracompressionbound"></a>

### `AroraCompressionBound`

[Исходник](../src/dl_complexity/bounds/compression.py#L74)

```text
AroraCompressionBound(delta: float = 0.05)
```

Upper-bound risk of compressed g using Theorem 2.1's proof.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Assumes iid samples and a data-independent decoder/alphabet/helper string.
The input certificate permits the empirical gamma-margin loss of original
f to upper-bound empirical classification error of compressed g. The raw
result may exceed one; it is never clipped or transferred to original f.
Selecting q/r/helper after seeing data requires extra accounting, not just
setting an independence flag. No layer-cushion/stability theorem is implied.
```

</details>

---

<a id="api-dl_complexity-bounds-inputs-aroracompressioninputs"></a>

### `AroraCompressionInputs`

[Исходник](../src/dl_complexity/bounds/inputs.py#L10)

```text
AroraCompressionInputs(
    empirical_margin_loss: torch.Tensor | float,
    n_observations: int,
    parameter_count: int,
    values_per_parameter: int,
    margin: float,
    max_logit_error: torch.Tensor | float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    certificate_kind: str = 'user_supplied_summary',
    compressed_state_id: str | None = None,
    provenance: dl_complexity.results.Provenance | None = None
)
```

A compression certificate and a fixed finite parameterization of g.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
``parameter_count`` is a predeclared upper budget q, not necessarily the
number of nonzero weights. Each of those q coordinates has at most r fixed
discrete values. Learned scales/codebooks/support indices must be included
in that parameterization; a serialized byte count is NOT a replacement.
The alphabet, decoder and helper string must be fixed independently of S.

``max_logit_error`` bounds the coordinatewise error on the ENTIRE training
sample. For margin threshold gamma we require error <= gamma/2, so the
multiclass margin cannot cross zero outside the gamma-margin error set.
Inputs can be explicitly user supplied or prepared from measured logits;
neither a boolean declaration nor this dataclass proves independence.

Args:
    empirical_margin_loss (Tensor or float): Fraction of sample margins <= gamma.
    n_observations (int): Full training sample size n.
    parameter_count (int): Predeclared finite coordinate budget q, not dense weights.
    values_per_parameter (int): Maximum finite coordinate alphabet size r.
    margin (float): Positive gamma threshold.
    max_logit_error (Tensor or float): Full-sample coordinate distortion <= gamma/2.
    helper_is_data_independent (bool): User declaration, not a measured fact.
    alphabet_is_data_independent (bool): User declaration for alphabet/q/r budget.
    certificate_kind (str): Origin label for measured or supplied summary.
    compressed_state_id (str, optional): Identity of the guaranteed predictor g.
    provenance (Provenance, optional): User-declared data/state identities.

Notes:
    O(1) construction/storage apart from referenced scalar tensors. This record
    cannot infer a finite predictor family from arbitrary serialized byte length.
```

</details>

---

<a id="api-dl_complexity-bounds-compression-arora_inputs_from_logits"></a>

### `arora_inputs_from_logits`

[Исходник](../src/dl_complexity/bounds/compression.py#L189)

```text
arora_inputs_from_logits(
    original_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    parameter_count: int,
    values_per_parameter: int,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None,
    provenance: dl_complexity.results.Provenance | None = None
) -> dl_complexity.bounds.inputs.AroraCompressionInputs
```

Measure a full-sample certificate, not a sampled/subset max-error proxy.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Logits must be [n, classes], finite and on the same device/dtype. Targets
index classes; ties and margins equal to the threshold count as margin
errors. Non-differentiable certificate statistics do not retain a graph.
This function cannot establish whether the declared class/helper was fixed
before these training observations were seen.

Args:
    original_logits (Tensor): Original predictor outputs [n, classes >= 2].
    compressed_logits (Tensor): Matching decoded predictor outputs.
    targets (LongTensor): Class-index labels [n].
    parameter_count (int): Independently fixed upper coefficient budget q.
    values_per_parameter (int): Independently fixed alphabet bound r.
    margin (float): Positive original-predictor margin threshold gamma.
    helper_is_data_independent (bool): Explicit fixed-helper declaration.
    alphabet_is_data_independent (bool): Explicit fixed q/r/alphabet declaration.
    compressed_state_id (str, optional): Declared identity of g.
    provenance (Provenance, optional): User-declared state/data identities.

Returns:
    AroraCompressionInputs: Empirical margin loss and maximum coordinate error.

Raises:
    InputError: If inputs are inconsistent or distortion exceeds gamma/2.

Notes:
    O(n*C) reductions and O(n*C) competitor/difference temporaries; no_grad.
    A subset of logits is not a full-sample certificate. Ties count as errors.
```

</details>

---

<a id="api-dl_complexity-bounds-compression-prepare_arora_compression"></a>

### `prepare_arora_compression`

[Исходник](../src/dl_complexity/bounds/compression.py#L293)

```text
prepare_arora_compression(
    original: torch.nn.modules.module.Module,
    compressed: torch.nn.modules.module.Module,
    data,
    *,
    parameter_count: int,
    values_per_parameter: int,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None
) -> dl_complexity.bounds.inputs.AroraCompressionInputs
```

Explicitly evaluate a certificate on all batches using isolated copies.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Scope: tensor inputs, class-index targets, deterministic eval logits for
ordinary small MLP/CNN models. This is preparation, never implicit training
or telemetry. Caller keeps a re-iterable source if other methods need it.
q/r describe a fixed discrete predictor family; codec bytes do not infer it.

Args:
    original (nn.Module): Original predictor f, copied before eval.
    compressed (nn.Module): Decoded candidate g, also copied.
    data (DataSource): Entire training sample as tensor/class-index batches.
    parameter_count (int): Declared fixed finite-coordinate budget q.
    values_per_parameter (int): Declared fixed alphabet upper bound r.
    margin (float): Positive margin threshold.
    helper_is_data_independent (bool): User-declared helper independence.
    alphabet_is_data_independent (bool): User-declared alphabet/budget independence.
    compressed_state_id (str, optional): Identity of g, not f.

Returns:
    AroraCompressionInputs: Measured sample certificate with explicit assumptions.

Raises:
    InputError: If batches or sample/certificate constraints are invalid.

Notes:
    Two model copies O(P_f+P_g), two full forward passes, and stored logits
    O(n*C) plus concatenation/difference temporaries. The implementation does
    not reduce the maximum online; it materializes all sample logits.
    CPU/model-parameter CUDA RNG is scoped; arbitrary external side effects
    or RNG on devices absent from those parameters are not universally isolated.
```

</details>

---

<a id="api-dl_complexity-bounds-compression-prepare_arora_projection"></a>

### `prepare_arora_projection`

[Исходник](../src/dl_complexity/bounds/compression.py#L380)

```text
prepare_arora_projection(
    original: torch.nn.modules.module.Module,
    projected,
    data,
    *,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None
) -> dl_complexity.bounds.inputs.AroraCompressionInputs
```

Certify an actual Matrix-Project artifact, deriving q/r from its code.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
The ENTIRE predictor schema is revalidated (no hooks, overrides, buffers,
structural changes or aliases), then g is freshly decoded from its bound
architecture and finite indices. Mutable instance forward is never used
for compressed logits. Independence of helper/alphabet remains an explicit
user declaration, not something logits can prove.

Args:
    original (nn.Module): Original predictor f.
    projected (ProjectedModel): Actual finite-index Matrix-Project artifact.
    data (DataSource): Entire training sample.
    margin (float): Positive margin threshold.
    helper_is_data_independent (bool): Explicit helper independence declaration.
    alphabet_is_data_independent (bool): Explicit alphabet/q/r independence declaration.
    compressed_state_id (str, optional): Declared identity of decoded g.

Returns:
    AroraCompressionInputs: Certificate bound to decoded indices and measured logits.

Raises:
    InputError: If the projection schema/state or sample certificate is invalid.

Notes:
    Adds ProjectedModel.decode costs to two full-sample forward passes and
    O(n*C) retained logits. q/r come from finite indices, never codec bytes.
    Independence remains an assumption; full noise-stability Theorem 4.1,
    layer cushions, rho_delta and a geometry-based budget are not implemented.
```

</details>

---


<a id="reference-diagnostics"></a>

## 📏 Диагностики весов

Импорт: `from dl_complexity.diagnostics import ...`.

[DiagnosticEstimator](#api-dl_complexity-diagnostics-base-diagnosticestimator) | [WeightInputs](#api-dl_complexity-diagnostics-inputs-weightinputs) | [WeightStatistics](#api-dl_complexity-diagnostics-weights-weightstatistics)

<a id="api-dl_complexity-diagnostics-base-diagnosticestimator"></a>

### `DiagnosticEstimator`

[Исходник](../src/dl_complexity/diagnostics/base.py#L6)

```text
DiagnosticEstimator(**config)
```

Share check/estimate for diagnostics without pretending they are code lengths.

<a id="api-dl_complexity-diagnostics-inputs-weightinputs"></a>

### `WeightInputs`

[Исходник](../src/dl_complexity/diagnostics/inputs.py#L11)

```text
WeightInputs(
    weights: Mapping[str, torch.Tensor],
    provenance: dl_complexity.results.Provenance | None = None
)
```

Declare named tensors whose structural/norm statistics should be measured.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    weights (Mapping[str, torch.Tensor]): Floating weight tensors; aliases allowed.
    provenance (Provenance, optional): Declared model identity.

Notes:
    The record retains references, not an independent checkpoint snapshot.
```

</details>

---

<a id="api-dl_complexity-diagnostics-weights-weightstatistics"></a>

### `WeightStatistics`

[Исходник](../src/dl_complexity/diagnostics/weights.py#L17)

```text
WeightStatistics(*, trainable_only=False, spectral=False)
```

Measure unique-tensor weight norms, sparsity and optional matrix spectral norms.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    trainable_only (bool): Context route selects only requires_grad parameters.
    spectral (bool): Include matrix operator-2 norms for exactly 2D tensors.

Notes:
    No primary scalar is forced. A flattened convolution filter's matrix norm
    would not automatically be the convolution operator norm.
```

</details>

---


<a id="reference-integrations"></a>

## 🔌 PyTorch-интеграции

Импорт: `from dl_complexity.integrations.torch import ...`.

[CategoricalLikelihood](#api-dl_complexity-integrations-torch-likelihood-categoricallikelihood) | [GaussianLikelihood](#api-dl_complexity-integrations-torch-likelihood-gaussianlikelihood) | [prepare_fit_statistics](#api-dl_complexity-integrations-torch-likelihood-prepare_fit_statistics) | [TrainingRun](#api-dl_complexity-integrations-torch-training-trainingrun) | [run_training](#api-dl_complexity-integrations-torch-training-run_training)

<a id="api-dl_complexity-integrations-torch-likelihood-categoricallikelihood"></a>

### `CategoricalLikelihood`

[Исходник](../src/dl_complexity/integrations/torch/likelihood.py#L18)

```text
CategoricalLikelihood()
```

Use a normalized categorical likelihood for one class label per row.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Notes:
    Input logits have shape [N,C]; targets have shape [N] and dtype long.
    Cross-entropy weights, soft labels and smoothing are not silently assumed.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-likelihood-categoricallikelihood-log_prob"></a>

#### `CategoricalLikelihood.log_prob`

[Исходник](../src/dl_complexity/integrations/torch/likelihood.py#L27)

```text
CategoricalLikelihood.log_prob(prediction, target)
```

Compute log_softmax(logits)[row, true_class] without batch reduction.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    prediction (torch.Tensor): Floating logits [N,C].
    target (torch.Tensor): Long class indices [N].

Returns:
    torch.Tensor: Natural-log probabilities [N] with prediction gradients.

Raises:
    InputError: Invalid shapes or target dtype.

Notes:
    O(N*C) time and log-softmax storage; gather returns N selected entries.
    Normalizing via log_softmax avoids exponentiating unshifted logits.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-likelihood-gaussianlikelihood"></a>

### `GaussianLikelihood`

[Исходник](../src/dl_complexity/integrations/torch/likelihood.py#L53)

```text
GaussianLikelihood(sigma: 'float' = 1.0)
```

Use independent Gaussian event coordinates with fixed known noise scale.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    sigma (float): Positive standard deviation, shared by all coordinates.

Notes:
    Includes normalization constants; MSE alone does not supply this density.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-likelihood-gaussianlikelihood-log_prob"></a>

#### `GaussianLikelihood.log_prob`

[Исходник](../src/dl_complexity/integrations/torch/likelihood.py#L71)

```text
GaussianLikelihood.log_prob(prediction, target)
```

Evaluate Normal log density and sum event axes, preserving observation axis.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    prediction (torch.Tensor): Mean values [N,...].
    target (torch.Tensor): Observations with exactly the same shape.

Returns:
    torch.Tensor: Total event log density per observation, shape [N], in nats.

Raises:
    InputError: Missing observation axis or incompatible shapes.

Notes:
    Per coordinate: -0.5*((y-mean)/sigma)**2-log(sigma)-0.5*log(2*pi).
    O(number of event coordinates) work/storage; no silent broadcasting.
    A density NLL may be negative and is not automatically a discrete code.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-likelihood-prepare_fit_statistics"></a>

### `prepare_fit_statistics`

[Исходник](../src/dl_complexity/integrations/torch/likelihood.py#L98)

```text
prepare_fit_statistics(context, *, data: 'DataSource', parameter_count='trainable')
```

Compute total log likelihood, declared parameter dimension and observed n.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext): Model and explicit likelihood; loss alone is insufficient.
    data (DataSource): Batches with pointwise log_prob output [N_batch].
    parameter_count (str or int): 'trainable', 'all' or explicit nonnegative k.

Returns:
    FitStatistics: Total natural log likelihood, k, n and declared provenance.

Raises:
    InputError: Missing likelihood, wrong batch/reduction, nonfinite likelihood
        or observed n disagreeing with declared n.

Notes:
    Counts unique Parameter objects, not effective statistical degrees of
    freedom; no optimization or MLE claim. Evaluates an isolated eval-mode copy,
    sums log_prob over observations and validates n after consumption.
    Time includes copying model state plus all forwards/likelihood reductions.
    Grad mode is preserved: enabled autograd can retain every batch's graph,
    not only the last batch. Use torch.no_grad() explicitly when appropriate.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-training-trainingrun"></a>

### `TrainingRun`

[Исходник](../src/dl_complexity/integrations/torch/training.py#L14)

```text
TrainingRun(
    model: 'Any',
    returned: 'Any',
    objective: 'str',
    config: 'Mapping[str, Any]',
    rng_seed: 'int',
    status: 'str' = 'completed'
)
```

Return one explicitly trained model and the callable's actual return value.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    model (nn.Module): Independent model copy modified by the recipe.
    returned: Original callable result, not automatically detached or serialized.
    objective (str): Declared objective, not verified by tracing Python code.
    config (Mapping): Recipe arguments, including a distinct callable seed if any.
    rng_seed (int): Outer Torch RNG seed.
    status (str): Completed for returned successful runs.

Notes:
    Unlike context.runs metadata, this returned object can retain arbitrary
    tensors/graphs through ``returned``. The caller controls their lifetime.
```

</details>

---

<a id="api-dl_complexity-integrations-torch-training-run_training"></a>

### `run_training`

[Исходник](../src/dl_complexity/integrations/torch/training.py#L39)

```text
run_training(context, *, seed: 'int')
```

Execute the attached recipe explicitly on an independent model copy.

<details>
<summary>Параметры, результат и ограничения</summary>

```text
Args:
    context (ModelContext): Must have a TrainingRecipe.
    seed (int): Outer Torch RNG seed, separate from recipe.config['seed'].

Returns:
    TrainingRun: Trained copy, callable return value and declared settings.

Raises:
    InputError: Recipe missing.
    Exception: Original training error, re-raised after recording failure.

Notes:
    Calls recipe.train(copy, **config), restoring scoped Torch RNG. Appends only
    TrainingRecord metadata to context.runs; source model remains unchanged.
    Work is O(model copy) plus user training cost; optimizer/data policy belongs
    to the callable. No telemetry or proof of the declared objective is added.
```

</details>

---


<a id="reference-errors"></a>

## ⚠️ Исключения

Импорт: `from dl_complexity.errors import ...`.

[InputError](#api-dl_complexity-errors-inputerror) | [PreparationRequired](#api-dl_complexity-errors-preparationrequired)

<a id="api-dl_complexity-errors-inputerror"></a>

### `InputError`

[Исходник](../src/dl_complexity/errors.py#L4)

Invalid, inconsistent or insufficient mathematical inputs.

<a id="api-dl_complexity-errors-preparationrequired"></a>

### `PreparationRequired`

[Исходник](../src/dl_complexity/errors.py#L8)

The caller must explicitly prepare the method's artifacts.
