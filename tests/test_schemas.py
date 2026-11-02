from datetime import datetime

import pytest
from pydantic import ValidationError

from ingest.schemas import KST, START_TEXT_FORMAT, Event, FrameResult


def _kst_pair(text: str) -> tuple[str, int]:
    ts = int(datetime.strptime(text, START_TEXT_FORMAT).replace(tzinfo=KST).timestamp())
    return text, ts


def test_frame_result_accepts_valid_payload() -> None:
    result = FrameResult(
        person_present=True,
        action="물 마시기",
        objects=["컵"],
        description="사람이 컵을 들고 물을 마시고 있다.",
        confidence=0.8,
        observed=True,
    )
    assert result.action == "물 마시기"


def test_frame_result_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FrameResult(
            person_present=True,
            action="식사",
            objects=[],
            description="식사 중이다.",
            confidence=1.5,
            observed=True,
        )


def test_frame_result_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        FrameResult(
            person_present=True,
            action="달리기",  # type: ignore[arg-type]
            objects=[],
            description="달리고 있다.",
            confidence=0.5,
            observed=True,
        )


def test_event_accepts_matching_kst_times() -> None:
    start_text, start_ts = _kst_pair("2026-11-01 03:00:00")
    event = Event(
        event_id="e1",
        video_id="v1",
        room="거실",
        start_ts=start_ts,
        end_ts=start_ts + 10,
        start_text=start_text,
        action="휴식",
        objects=["소파"],
        description="소파에 앉아 있다.",
        confidence=0.7,
        observed=True,
    )
    assert event.clip_kept is False


def test_event_rejects_mismatched_start_text_and_ts() -> None:
    start_text, start_ts = _kst_pair("2026-11-01 03:00:00")
    with pytest.raises(ValidationError):
        Event(
            event_id="e1",
            video_id="v1",
            room="주방",
            start_ts=start_ts + 60,
            end_ts=start_ts + 70,
            start_text=start_text,
            action="식사",
            objects=[],
            description="식사 중이다.",
            confidence=0.6,
            observed=True,
        )


def test_event_jsonl_roundtrip() -> None:
    start_text, start_ts = _kst_pair("2026-11-01 12:30:00")
    original = Event(
        event_id="e2",
        video_id="v2",
        room="현관",
        start_ts=start_ts,
        end_ts=start_ts + 5,
        start_text=start_text,
        action="출입",
        objects=["문"],
        description="현관문을 열고 있다.",
        confidence=0.9,
        observed=True,
        clip_kept=False,
    )
    restored = Event.from_jsonl_line(original.to_jsonl_line())
    assert restored == original
