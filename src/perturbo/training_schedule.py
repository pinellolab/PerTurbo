from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingScheduleRequest:
    shared_steps: int | None = None
    shared_epochs: int | None = None
    control_steps: int | None = None
    beta_steps: int | None = None
    control_epochs: int | None = None
    beta_epochs: int | None = None

    def resolve_stage_steps(
        self,
        *,
        stage: str,
        num_cells: int,
        minibatch_size: int | None,
    ) -> int:
        if stage not in {"control", "beta"}:
            raise ValueError(f"Unknown training stage: {stage}")
        if num_cells < 1:
            raise ValueError("num_cells must be >= 1.")

        if self.shared_steps is not None:
            return self.shared_steps
        if self.shared_epochs is not None:
            return _steps_from_epochs(self.shared_epochs, num_cells=num_cells, minibatch_size=minibatch_size)
        if stage == "control":
            if self.control_steps is not None:
                return self.control_steps
            if self.control_epochs is not None:
                return _steps_from_epochs(self.control_epochs, num_cells=num_cells, minibatch_size=minibatch_size)
        else:
            if self.beta_steps is not None:
                return self.beta_steps
            if self.beta_epochs is not None:
                return _steps_from_epochs(self.beta_epochs, num_cells=num_cells, minibatch_size=minibatch_size)
        raise RuntimeError("Training schedule request is unresolved.")


def _require_positive(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be >= 1.")


def _steps_from_epochs(epochs: int, *, num_cells: int, minibatch_size: int | None) -> int:
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    effective_minibatch = None
    if minibatch_size not in (None, 0):
        effective_minibatch = int(minibatch_size)
        if effective_minibatch < 1:
            raise ValueError("minibatch_size must be >= 1 when provided.")
        if effective_minibatch >= num_cells:
            effective_minibatch = None
    steps_per_epoch = 1 if effective_minibatch is None else math.ceil(num_cells / effective_minibatch)
    return int(epochs) * steps_per_epoch


def resolve_training_schedule(
    *,
    shared_steps: int | None = None,
    shared_epochs: int | None = None,
    control_steps: int | None = None,
    beta_steps: int | None = None,
    control_epochs: int | None = None,
    beta_epochs: int | None = None,
    default_shared_steps: int | None = None,
    default_shared_epochs: int | None = None,
    default_control_steps: int | None = None,
    default_beta_steps: int | None = None,
) -> TrainingScheduleRequest:
    _require_positive("shared_steps", shared_steps)
    _require_positive("shared_epochs", shared_epochs)
    _require_positive("control_steps", control_steps)
    _require_positive("beta_steps", beta_steps)
    _require_positive("control_epochs", control_epochs)
    _require_positive("beta_epochs", beta_epochs)
    _require_positive("default_shared_steps", default_shared_steps)
    _require_positive("default_shared_epochs", default_shared_epochs)
    _require_positive("default_control_steps", default_control_steps)
    _require_positive("default_beta_steps", default_beta_steps)

    step_mode = any(value is not None for value in (shared_steps, control_steps, beta_steps))
    epoch_mode = any(value is not None for value in (shared_epochs, control_epochs, beta_epochs))
    if step_mode and epoch_mode:
        raise ValueError("Step-based and epoch-based scheduling cannot be mixed.")

    if shared_steps is not None and (control_steps is not None or beta_steps is not None):
        raise ValueError("Shared step scheduling cannot be mixed with stage-specific step scheduling.")
    if shared_epochs is not None and (control_epochs is not None or beta_epochs is not None):
        raise ValueError("Shared epoch scheduling cannot be mixed with stage-specific epoch scheduling.")

    if (control_steps is None) ^ (beta_steps is None):
        raise ValueError("control_steps and beta_steps must be provided together.")
    if (control_epochs is None) ^ (beta_epochs is None):
        raise ValueError("control_epochs and beta_epochs must be provided together.")

    if not step_mode and not epoch_mode:
        default_modes = sum(
            value is not None
            for value in (
                default_shared_steps,
                default_shared_epochs,
                default_control_steps,
                default_beta_steps,
            )
        )
        if default_modes == 0:
            raise ValueError("No training schedule was provided.")
        if default_modes == 2 and default_control_steps is not None and default_beta_steps is not None:
            return TrainingScheduleRequest(control_steps=default_control_steps, beta_steps=default_beta_steps)
        if default_modes != 1:
            raise ValueError("Default training schedule configuration is ambiguous.")
        if default_shared_steps is not None:
            return TrainingScheduleRequest(shared_steps=default_shared_steps)
        if default_shared_epochs is not None:
            return TrainingScheduleRequest(shared_epochs=default_shared_epochs)
        raise ValueError("Incomplete default training schedule configuration.")

    return TrainingScheduleRequest(
        shared_steps=shared_steps,
        shared_epochs=shared_epochs,
        control_steps=control_steps,
        beta_steps=beta_steps,
        control_epochs=control_epochs,
        beta_epochs=beta_epochs,
    )


__all__ = [
    "TrainingScheduleRequest",
    "resolve_training_schedule",
]
