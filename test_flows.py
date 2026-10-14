from app.orchestration.schedules import SUGGESTED_SCHEDULES


def test_suggested_schedules_present() -> None:
    assert "training_flow" in SUGGESTED_SCHEDULES
    assert "monitoring_flow" in SUGGESTED_SCHEDULES
