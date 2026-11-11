Кодовые длины и prequential-протоколы
=====================================

Импорт: ``from dl_complexity.coding import ...``.
Исполняемые примеры и объяснения: :doc:`../prequential` и :doc:`../demo`.
Оценщики принимают подготовленный журнал; функции ``prepare_*`` явно
выполняют обучение через callbacks. Все записи журнала хранят NLL в натах.

Базовый контракт и входы
------------------------

.. automodule:: dl_complexity.coding.base
   :members:
   :show-inheritance:

.. automodule:: dl_complexity.coding.inputs
   :members:

Двухчастный код
---------------

.. automodule:: dl_complexity.coding.two_part
   :members:
   :show-inheritance:

Три оценщика
------------

``check`` и ``estimate`` наследуются от общего
:class:`~dl_complexity.base.ComplexityEstimator`.

.. automodule:: dl_complexity.coding.prequential.blockwise
   :members:
   :show-inheritance:

.. automodule:: dl_complexity.coding.prequential.online
   :members:
   :show-inheritance:

.. automodule:: dl_complexity.coding.prequential.replay
   :members:
   :show-inheritance:

Явная подготовка
----------------

.. automodule:: dl_complexity.coding.prequential.protocol
   :members: prepare_blockwise, prepare_online, prepare_replay

Калибровка
----------

.. automodule:: dl_complexity.coding.prequential.calibration
   :members:
