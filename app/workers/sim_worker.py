"""Runs one blocking SimService.run() call off the UI thread.

The whole t_span is integrated in a single solve_ivp call (see APP_STACK_SKETCH.md
§4), so this worker has exactly one step: run, then emit finished or failed.
There is no progress/pause hook yet -- that arrives with a stepwise integrator.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from app.services.sim_service import SimService
from simulator2.io_types import ExtendedTrainScenario


class SimWorker(QObject):
    finished = Signal(object)  # TensorSimulationResult
    failed = Signal(str)

    def __init__(self, svc: SimService, scenario: ExtendedTrainScenario) -> None:
        super().__init__()
        self._svc = svc
        self._scenario = scenario

    @Slot()
    def run(self) -> None:
        try:
            result = self._svc.run(self._scenario)
        except Exception as exc:  # e.g. solve_ivp RuntimeError(sol.message)
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)
