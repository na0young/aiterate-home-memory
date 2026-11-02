from datetime import datetime, timedelta

from ingest.mini_demo import merge_frames, to_event
from ingest.schemas import FrameResult

T0 = datetime(2026, 10, 27, 8, 10, 0)


def _at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _frame(action: str = "물 마시기", **kwargs) -> FrameResult:
    base = {
        "person_present": True,
        "action": action,
        "objects": ["컵"],
        "description": "물을 마신다.",
        "confidence": 0.8,
        "observed": True,
    }
    return FrameResult(**{**base, **kwargs})


def test_merge_joins_same_action() -> None:
    frames = [(_at(0), _frame(objects=["컵"])), (_at(1), _frame(objects=["물병"]))]
    segments = merge_frames(frames)
    assert len(segments) == 1
    assert segments[0].start == _at(0)
    assert segments[0].end == _at(1)
    assert segments[0].objects == {"컵", "물병"}


def test_merge_splits_when_action_changes() -> None:
    frames = [(_at(0), _frame("물 마시기")), (_at(1), _frame("식사"))]
    assert [segment.action for segment in merge_frames(frames)] == ["물 마시기", "식사"]


def test_merge_breaks_when_person_absent() -> None:
    frames = [
        (_at(0), _frame()),
        (_at(1), _frame(person_present=False)),
        (_at(2), _frame()),
    ]
    assert len(merge_frames(frames)) == 2


def test_merge_observed_is_false_if_any_frame_is_a_guess() -> None:
    frames = [(_at(0), _frame()), (_at(1), _frame(observed=False))]
    assert merge_frames(frames)[0].observed is False


def test_to_event_makes_valid_common_event() -> None:
    frames = [(_at(0), _frame(confidence=0.6)), (_at(2), _frame(confidence=1.0))]
    segment = merge_frames(frames)[0]
    event = to_event("E1", "p4_08_living", "거실", segment, gap_sec=2)
    assert event.start_text == "2026-10-27 08:10:00"
    assert event.end_ts - event.start_ts == 4  # 마지막 프레임(2초) + 프레임 간격(2초)
    assert event.confidence == 0.8
    assert event.room == "거실"
