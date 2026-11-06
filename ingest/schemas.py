"""팀 공통 이벤트/프레임 형식. 파이프라인 로직은 넣지 않는다."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

KST = ZoneInfo("Asia/Seoul")

Action = Literal["약 복용", "물 마시기", "식사", "택배 수령", "출입", "기상", "휴식", "기타 활동"]
Room = Literal["거실", "주방", "현관", "안방"]
ACTIONS: list[str] = list(get_args(Action))
ROOMS: list[str] = list(get_args(Room))
FILENAME_ROOM_MAP: dict[str, str] = {
    "living": "거실",
    "kitchen": "주방",
    "entrance": "현관",
    "bedroom": "안방",
}

START_TEXT_FORMAT = "%Y-%m-%d %H:%M:%S"


class FrameResult(BaseModel):
    """VLM이 프레임 1장에서 반환하는 값."""

    person_present: bool
    action: Action
    objects: list[str]
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    observed: bool


class FrameRecord(BaseModel):
    """프레임 1장의 처리 기록. frames/<영상>.jsonl 한 줄이며 팀 공통 형식은 아니다."""

    video_id: str
    room: Room
    video_start: str  # 영상이 시작된 가상 시각 'YYYY-MM-DD HH:MM'
    t_sec: float  # 영상 안에서 이 프레임의 시각(초)
    ok: bool  # False면 VLM 응답이 형식에 끝내 맞지 않아 결과가 없음
    attempts: int  # 시도 횟수 (1이면 재시도 없이 성공)
    elapsed_sec: float
    result: FrameResult | None = None


class Event(BaseModel):
    """검색·서버에서 쓰는 팀 공통 이벤트 형식."""

    event_id: str
    video_id: str
    room: Room
    start_ts: int
    end_ts: int
    start_text: str
    action: Action
    objects: list[str]
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    observed: bool
    clip_kept: bool = False

    @field_validator("start_text")
    @classmethod
    def validate_start_text_format(cls, value: str) -> str:
        try:
            datetime.strptime(value, START_TEXT_FORMAT)
        except ValueError as exc:
            raise ValueError(
                f"start_text는 {START_TEXT_FORMAT} 형식이어야 합니다: {value!r}"
            ) from exc
        return value

    @model_validator(mode="after")
    def validate_start_text_matches_ts(self) -> Event:
        parsed = datetime.strptime(self.start_text, START_TEXT_FORMAT).replace(tzinfo=KST)
        expected_ts = int(parsed.timestamp())
        if expected_ts != self.start_ts:
            raise ValueError(
                "start_text와 start_ts가 KST 기준으로 일치하지 않습니다. "
                f"start_text={self.start_text!r} → {expected_ts}, start_ts={self.start_ts}"
            )
        return self

    def to_jsonl_line(self) -> str:
        """한 줄 JSON 문자열을 반환한다."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> Event:
        """JSONL 한 줄을 Event로 복원한다."""
        return cls.model_validate_json(line.strip())
