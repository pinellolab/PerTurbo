from __future__ import annotations

import pytest

from perturbo.training_schedule import resolve_training_schedule


def test_resolve_training_schedule_uses_full_batch_epochs_as_dataset_passes() -> None:
    schedule = resolve_training_schedule(shared_epochs=3)

    assert schedule.resolve_stage_steps(stage="control", num_cells=10, minibatch_size=None) == 3
    assert schedule.resolve_stage_steps(stage="beta", num_cells=10, minibatch_size=10) == 3


def test_resolve_training_schedule_converts_minibatch_epochs_to_steps() -> None:
    schedule = resolve_training_schedule(control_epochs=2, beta_epochs=3)

    assert schedule.resolve_stage_steps(stage="control", num_cells=5, minibatch_size=2) == 6
    assert schedule.resolve_stage_steps(stage="beta", num_cells=7, minibatch_size=3) == 9


def test_resolve_training_schedule_rejects_mixed_step_and_epoch_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        resolve_training_schedule(shared_steps=5, shared_epochs=2)


def test_resolve_training_schedule_requires_complete_split_pairs() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        resolve_training_schedule(control_steps=5)

    with pytest.raises(ValueError, match="must be provided together"):
        resolve_training_schedule(beta_epochs=3)
