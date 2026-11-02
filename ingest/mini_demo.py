"""영상 -> 이벤트 변환(ingest)과 간단한 질의응답 데모.

저장소 루트에서 실행한다.
python -m ingest.mini_demo ingest p1_03_living.MOV --room 거실 --start "2026-10-27 08:10"
python -m ingest.mini_demo show
python -m ingest.mini_demo ask "오늘 약 드셨어?"

설정은 ingest/config.py, 데이터 형식은 ingest/schemas.py에 있다.
이 파일은 프레임 1장 VLM 호출, 영상 처리, 이벤트 보기와 질문을 담당한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from ollama import Client
from pydantic import ValidationError

from ingest.config import (
    EMB_MODEL,
    EVENTS_DB,
    FRAME_GAP_SEC,
    FRAME_MAX_SIDE,
    LLM_MODEL,
    MAX_VLM_RETRIES,
    OLLAMA_HOST,
    VLM_MODEL,
)
from ingest.schemas import ACTIONS, KST, ROOMS, START_TEXT_FORMAT, Event, FrameResult

logger = logging.getLogger(__name__)

TOP_K = 5

# gemma4:e4b(MLX)는 Ollama format(structured output)을 지원하지 않는다.
# 그래서 format은 보내지 않고, 응답 JSON을 파싱한 뒤 pydantic으로 검증한다.
SYSTEM_PROMPT = (
    "너는 홈캠 프레임만 보고 JSON 객체 하나만 출력하는 관찰자다.\n"
    "- 마크다운 코드블록 없이 JSON만 출력한다.\n"
    "- 화면에 직접 보이는 사실만 적는다. 추측하지 않는다.\n"
    "- 확실하지 않으면 action은 '기타 활동'이고 observed는 false다.\n"
    f"- action은 영어가 아니라 다음 한국어 값 중 하나만 고른다: {', '.join(ACTIONS)}\n"
    "- 사람이 보이지 않으면 person_present는 false다.\n"
    "- 사람은 '사람'이라고만 쓴다. 성별, 나이, 얼굴 생김새, 이름, "
    "화면이나 종이에 적힌 글자는 쓰지 않는다.\n"
    "- description은 한국어 한 문장이다.\n"
    "- objects는 화면에 보이는 물체 이름만 한국어로 1~3개 적는다.\n"
    "행동 판단 기준:\n"
    "- 알약을 입에 넣거나, 약통·알약이 보이는 상태에서 컵으로 마시면 '약 복용'\n"
    "- 약통이 없고 컵으로 마시기만 하면 '물 마시기'\n"
    "- 약통을 열거나 들고만 있고 먹는 모습이 없으면 '기타 활동' (설명에 본 것만 쓰기)\n"
    "- 휴대폰을 귀에 대고 있으면 '기타 활동' (설명에 통화라고 쓰기)\n"
    "- 노트북·컴퓨터는 배경일 뿐이니, 손에 든 물건과 손의 움직임을 먼저 본다\n"
    "- 키: person_present(bool), action(str), objects(list[str]), "
    "description(str), confidence(0~1 float), observed(bool)"
)

USER_PROMPT = "이 프레임에서 사람과 활동을 관찰하고 JSON만 출력하라."
RETRY_PROMPT = (
    f"이전 답은 스키마에 맞지 않았다. action은 {ACTIONS} 중 하나만, "
    "JSON만 출력하고 다른 텍스트는 쓰지 마라."
)

ANSWER_RULES = (
    "당신은 집 안 기록을 보고 답하는 도우미입니다. "
    "아래 [기록]만 근거로 한국어로 짧게 답하세요. "
    "근거로 쓴 기록 번호를 [E번호] 형태로 붙이세요. "
    "기록에 근거가 없으면 '기록에서 확인되지 않습니다'라고만 답하고 추측하지 마세요."
)

_client = Client(host=OLLAMA_HOST)


class VLMResponseError(RuntimeError):
    """Ollama 응답이 FrameResult 스키마에 맞지 않을 때 발생한다."""


@dataclass
class Segment:
    """같은 행동이 이어진 프레임 묶음. 시각은 한국 시간(KST) 기준의 naive datetime이다."""

    action: str
    description: str
    start: datetime
    end: datetime
    objects: set[str]
    confidences: list[float]
    observed: bool


# ---------------------------------------------------------------------------
# 프레임 1장 -> FrameResult (VLM 호출)
# ---------------------------------------------------------------------------


def describe_frame(image: str | Path | np.ndarray) -> FrameResult:
    """이미지 경로 또는 OpenCV BGR 배열을 FrameResult로 변환한다."""
    jpeg_bytes = _encode_jpeg(_resize_long_side(_load_bgr(image), FRAME_MAX_SIDE))
    user_prompt = USER_PROMPT
    attempts = MAX_VLM_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [jpeg_bytes]},
        ]
        try:
            return FrameResult.model_validate_json(_extract_json_object(_chat(VLM_MODEL, messages)))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            user_prompt = f"{USER_PROMPT}\n{RETRY_PROMPT}"
            logger.warning("VLM JSON 검증 실패 (%s/%s): %s", attempt + 1, attempts, exc)
    raise VLMResponseError(
        f"Ollama 응답이 FrameResult 스키마에 맞지 않습니다. {MAX_VLM_RETRIES}회 재시도 후 실패."
    ) from last_error


def _chat(model: str, messages: list[dict]) -> str:
    """Ollama chat 호출. 같은 입력엔 같은 답이 나오게 temperature 0, thinking은 끈다."""
    response = _client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0},
        think=False,
    )
    content = response.message.content
    if not content:
        raise ValueError("Ollama 응답 content가 비어 있습니다.")
    return content


def _load_bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("빈 이미지 배열입니다.")
        return image
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
    loaded = cv2.imread(str(path))
    if loaded is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return loaded


def _resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("JPEG 인코딩에 실패했습니다.")
    return buffer.tobytes()


def _extract_json_object(text: str) -> str:
    """응답 문자열에서 JSON 객체만 잘라낸다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON 객체를 찾지 못했습니다: {text[:200]!r}")
    payload = stripped[start : end + 1]
    json.loads(payload)
    return payload


# ---------------------------------------------------------------------------
# 프레임들 -> 이벤트 (병합, 공통 Event로 변환)
# ---------------------------------------------------------------------------


def merge_frames(frames: list[tuple[datetime, FrameResult]]) -> list[Segment]:
    """같은 행동이 이어지는 프레임을 구간 하나로 묶는다. 사람이 없는 프레임은 구간을 끊는다."""
    # TODO: 짧게 끊긴 구간 흡수, 설명은 최빈값으로
    segments: list[Segment] = []
    current: Segment | None = None
    for time_at, result in frames:
        if not result.person_present:
            current = None
            continue
        if current is not None and current.action == result.action:
            current.end = time_at
            current.objects |= set(result.objects)
            current.confidences.append(result.confidence)
            current.observed = current.observed and result.observed
            continue
        current = Segment(
            action=result.action,
            description=result.description,
            start=time_at,
            end=time_at,
            objects=set(result.objects),
            confidences=[result.confidence],
            observed=result.observed,
        )
        segments.append(current)
    return segments


def to_event(event_id: str, video_id: str, room: str, segment: Segment, gap_sec: float) -> Event:
    """구간 하나를 팀 공통 Event로 바꾼다. 끝 시각은 마지막 프레임 + 프레임 간격이다."""
    start = segment.start.replace(microsecond=0)
    end = segment.end.replace(microsecond=0) + timedelta(seconds=max(1, round(gap_sec)))
    return Event(
        event_id=event_id,
        video_id=video_id,
        room=room,
        start_ts=_to_ts(start),
        end_ts=_to_ts(end),
        start_text=start.strftime(START_TEXT_FORMAT),
        action=segment.action,
        objects=sorted(segment.objects),
        description=segment.description,
        confidence=round(mean(segment.confidences), 2),
        observed=segment.observed,
    )


def _to_ts(moment: datetime) -> int:
    return int(moment.replace(tzinfo=KST).timestamp())


# ---------------------------------------------------------------------------
# 저장, 보기, 영상 처리, 질문
# ---------------------------------------------------------------------------


def load_events() -> list[Event]:
    if not EVENTS_DB.exists():
        return []
    raw = json.loads(EVENTS_DB.read_text(encoding="utf-8"))
    try:
        return [Event.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise SystemExit(f"{EVENTS_DB.name}이 옛 형식입니다. 지우고 다시 ingest하세요.") from exc


def save_events(events: list[Event]) -> None:
    payload = [event.model_dump() for event in events]
    EVENTS_DB.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def event_line(event: Event) -> str:
    end = datetime.fromtimestamp(event.end_ts, KST)
    objects = ", ".join(event.objects) or "-"
    return (
        f"[{event.event_id}] {event.start_text}~{end:%H:%M:%S} {event.room} | "
        f"{event.action} | {event.description} | 물건: {objects}"
    )


def ingest_video(video: Path, room: str, start: datetime, gap_sec: float) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video} (파일 이름을 영문으로 바꿔 보세요)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * gap_sec))

    frames: list[tuple[datetime, FrameResult]] = []
    times: list[float] = []
    failed = 0
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        moment = start + timedelta(seconds=idx / fps)
        began = time.perf_counter()
        try:
            result = describe_frame(frame)
        except VLMResponseError:
            failed += 1
            print(f"  {moment:%H:%M:%S}  (건너뜀: 형식 오류)")
            continue
        elapsed = time.perf_counter() - began
        times.append(elapsed)
        frames.append((moment, result))
        print(f"  {moment:%H:%M:%S}  {result.action:<6} {result.description}  ({elapsed:.1f}초)")
    cap.release()

    events = load_events()
    segments = merge_frames(frames)
    new_events = [
        to_event(f"E{len(events) + number}", video.stem, room, segment, gap_sec)
        for number, segment in enumerate(segments, start=1)
    ]
    save_events(events + new_events)

    print(f"\n프레임 {len(frames)}장 -> 이벤트 {len(new_events)}건 저장")
    print(f"누적 {len(events) + len(new_events)}건 ({EVENTS_DB.name})")
    if failed:
        print(f"형식 오류로 건너뛴 프레임 {failed}장")
    if times:
        print(f"프레임당 평균 {mean(times):.1f}초")
    if len(times) > 1:
        print(f"첫 장 빼고 평균 {mean(times[1:]):.1f}초")


def show() -> None:
    events = load_events()
    print("\n".join(event_line(event) for event in events) or "저장된 이벤트가 없습니다.")


def ask(question: str, now: datetime) -> None:
    events = load_events()
    if not events:
        raise SystemExit("저장된 이벤트가 없습니다. 먼저 ingest를 실행하세요.")
    began = time.perf_counter()
    texts = [event_line(event) for event in events]
    embedded = _client.embed(model=EMB_MODEL, input=[question, *texts])["embeddings"]
    vecs = np.array(embedded)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = vecs[1:] @ vecs[0]
    top = np.argsort(-scores)[:TOP_K]

    records = "\n".join(texts[i] for i in sorted(top))
    prompt = f"현재 시각: {now:%Y-%m-%d %H:%M}\n\n[기록]\n{records}\n\n[질문] {question}"
    messages = [
        {"role": "system", "content": ANSWER_RULES},
        {"role": "user", "content": prompt},
    ]
    answer = _chat(LLM_MODEL, messages).strip()
    print(f"\n질문: {question}\n답변: {answer}\n")
    print("검색된 근거 (유사도 순)")
    for i in top:
        print(f"  {scores[i]:.2f}  {texts[i]}")
    print(f"\n응답 시간 {time.perf_counter() - began:.1f}초")


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="픽셀 없는 홈 메모리 미니 데모")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_cmd = sub.add_parser("ingest", help="영상을 이벤트로 바꿔 저장")
    ingest_cmd.add_argument("video")
    ingest_cmd.add_argument("--room", choices=ROOMS, default="거실")
    ingest_cmd.add_argument(
        "--start",
        type=parse_time,
        default=datetime.now().replace(second=0, microsecond=0),
        help='영상이 시작된 시각, 예: "2026-10-11 08:00"',
    )
    ingest_cmd.add_argument(
        "--gap",
        type=float,
        default=FRAME_GAP_SEC,
        help="프레임 추출 간격(초)",
    )

    sub.add_parser("show", help="저장된 이벤트 보기")

    ask_cmd = sub.add_parser("ask", help="질문하기")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument(
        "--now",
        type=parse_time,
        default=datetime.now(),
        help="'오늘'의 기준 시각",
    )

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest_video(Path(args.video), args.room, args.start, args.gap)
    elif args.cmd == "show":
        show()
    else:
        ask(args.question, args.now)


if __name__ == "__main__":
    main()
