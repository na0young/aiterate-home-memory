"""영상 -> 프레임 기록 -> 이벤트 JSONL.

저장소 루트에서 실행한다. 방과 영상 시작 시각은 data/scenario.csv에서 영상 이름으로 찾는다.
python -m ingest.pipeline data/videos/p1_03_living.MOV   # 영상 하나
python -m ingest.pipeline                                # data/videos/ 전부 (p3 평가용은 제외)
python -m ingest.pipeline --gap 3 --run try2             # 프레임 간격을 바꿔 비교
python -m ingest.pipeline --run try1 --merge-only        # VLM 없이 이벤트만 다시 만든다

결과는 data/outputs/<run>/ 에 쌓인다. (--run을 안 주면 default)
<영상>.frames.jsonl (프레임별 기록), events.jsonl (그 run의 모든 영상의 이벤트)
프레임 이미지는 메모리에서만 쓰고 파일로 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from ingest.config import (
    FRAME_GAP_SEC,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SCENARIO_CSV,
    TEST_PERSON,
    VIDEO_DIR,
)
from ingest.schemas import KST, START_TEXT_FORMAT, Event, FrameRecord
from ingest.vlm import VLMResponseError, describe_frame

VIDEO_EXTENSIONS = {".mp4", ".mov"}  # 소문자로 비교한다 (.MOV도 해당)
VIDEO_ID_PATTERN = re.compile(r"^p(?P<person>\d)_(?P<scene>\d{2})_")
START_FORMAT = "%Y-%m-%d %H:%M"
FRAMES_SUFFIX = ".frames.jsonl"  # 프레임 기록 파일 이름의 끝 (events.jsonl과 구분)


# ---------------------------------------------------------------------------
# 시나리오 (방과 영상 시작 시각은 영상에서 알 수 없어서 data/scenario.csv에서 가져온다)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """scenario.csv의 한 줄. start는 한국 시간(KST) 기준의 naive datetime이다.

    scenario.csv에는 정답(action, objects, description)도 들어 있지만, 여기서는 읽지 않는다.
    """

    video_id: str
    start: datetime
    room: str


def load_scenario(path: Path) -> dict[str, Scene]:
    """scenario.csv를 {video_id: Scene}으로 읽는다."""
    if not path.is_file():
        raise FileNotFoundError(f"시나리오 파일이 없습니다: {path}")
    scenario: dict[str, Scene] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            start = datetime.strptime(f"{row['date']} {row['start']}", START_FORMAT)
            scenario[row["video_id"]] = Scene(row["video_id"], start, row["room"])
    return scenario


# ---------------------------------------------------------------------------
# 프레임 추출, VLM 호출
# ---------------------------------------------------------------------------


def iter_frames(video_path: Path, gap_sec: float) -> Iterator[tuple[float, np.ndarray]]:
    """gap_sec 간격으로 (영상 안의 시각(초), 프레임)을 하나씩 꺼낸다.

    영상을 처음부터 순서대로 읽으면서 필요한 프레임만 변환한다. (건너뛰기보다 정확하다)
    """
    if gap_sec <= 0:
        raise ValueError(f"프레임 간격은 0보다 커야 합니다: {gap_sec}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path} "
            "(파일 이름을 영문으로 바꾸거나 mp4로 변환해 보세요)"
        )
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        index = 0
        taken = 0
        while cap.grab():
            t_sec = index / fps
            if t_sec + 1e-6 >= taken * gap_sec:
                ok, frame = cap.retrieve()
                if ok:
                    yield t_sec, frame
                taken += 1
            index += 1
    finally:
        cap.release()


def describe_video(video_path: Path, gap_sec: float, scene: Scene) -> list[FrameRecord]:
    """영상의 프레임을 하나씩 VLM에 넣고 FrameRecord 목록을 돌려준다.

    VLM 응답이 형식에 끝내 맞지 않은 프레임은 ok=False로 기록하고 계속 진행한다.
    """
    records: list[FrameRecord] = []
    for t_sec, frame in iter_frames(video_path, gap_sec):
        started = time.perf_counter()
        try:
            result, attempts = describe_frame(frame)
        except VLMResponseError as exc:
            result, attempts = None, exc.attempts
        record = FrameRecord(
            video_id=video_path.stem,
            room=scene.room,
            video_start=scene.start.strftime(START_FORMAT),
            t_sec=round(t_sec, 2),
            ok=result is not None,
            attempts=attempts,
            elapsed_sec=round(time.perf_counter() - started, 3),
            result=result,
        )
        records.append(record)
        print_frame(record)
    return records


def print_frame(record: FrameRecord) -> None:
    label = f"{record.video_id} {record.t_sec:6.1f}초"
    result = record.result
    if result is None:
        print(f"  {label}  [형식 오류로 건너뜀] ({record.elapsed_sec:.1f}초)")
        return
    action = result.action if result.person_present else "사람 없음"
    print(f"  {label}  {action:<6} {result.description}  ({record.elapsed_sec:.1f}초)")


# ---------------------------------------------------------------------------
# 병합, 이벤트 변환
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """같은 행동이 이어진 프레임 묶음. 시각은 영상 안의 초 단위다."""

    action: str
    description: str
    start_sec: float
    end_sec: float
    objects: set[str]
    confidences: list[float]
    observed: bool


def merge_records(records: list[FrameRecord]) -> list[Segment]:
    """같은 행동이 이어지는 프레임을 구간 하나로 묶는다.

    - 사람이 없는 프레임은 구간을 끊는다.
    - 형식 오류로 결과가 없는 프레임은 건너뛰고, 구간은 끊지 않는다.
    """
    # TODO: 짧게 끊긴 구간 흡수, 여러 프레임을 묶어 판단
    segments: list[Segment] = []
    current: Segment | None = None
    for record in sorted(records, key=lambda r: r.t_sec):
        result = record.result
        if not record.ok or result is None:
            continue
        if not result.person_present:
            current = None
            continue
        if current is not None and current.action == result.action:
            current.end_sec = record.t_sec
            current.objects |= set(result.objects)
            current.confidences.append(result.confidence)
            current.observed = current.observed and result.observed
            continue
        current = Segment(
            action=result.action,
            description=result.description,
            start_sec=record.t_sec,
            end_sec=record.t_sec,
            objects=set(result.objects),
            confidences=[result.confidence],
            observed=result.observed,
        )
        segments.append(current)
    return segments


def event_id_prefix(video_id: str) -> str:
    """정답 파일과 같은 이벤트 번호 앞부분. 'p1_03_living' -> 'E1003' (이름이 다르면 'E_이름')"""
    match = VIDEO_ID_PATTERN.match(video_id)
    if match is None:
        return f"E_{video_id}"
    return f"E{match['person']}{int(match['scene']):03d}"


def to_events(
    video_id: str,
    room: str,
    video_start: datetime,
    segments: list[Segment],
    gap_sec: float,
) -> list[Event]:
    """구간을 팀 공통 Event로 바꾼다. 끝 시각은 마지막 프레임 + 프레임 간격이다.

    한 영상에서 이벤트가 여러 개면 E1003, E1003-2처럼 번호를 붙인다.
    """
    prefix = event_id_prefix(video_id)
    gap = max(1, round(gap_sec))
    events: list[Event] = []
    for number, segment in enumerate(segments, start=1):
        start = video_start + timedelta(seconds=round(segment.start_sec))
        end = video_start + timedelta(seconds=round(segment.end_sec) + gap)
        events.append(
            Event(
                event_id=prefix if number == 1 else f"{prefix}-{number}",
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
        )
    return events


def _to_ts(moment: datetime) -> int:
    return int(moment.replace(tzinfo=KST).timestamp())


def infer_gap(records: list[FrameRecord]) -> float:
    """프레임 기록의 시각 차이로 추출 간격을 알아낸다. (기록이 1장뿐이면 기본값)"""
    times = sorted(r.t_sec for r in records)
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    return median(diffs) if diffs else FRAME_GAP_SEC


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def select_videos(explicit: list[str], scenario: dict[str, Scene]) -> list[Path]:
    """처리할 영상을 고른다. 시나리오에 있는 영상만 처리한다.

    영상을 직접 지정하지 않으면 data/videos/ 전부를 보되, 최종 평가용(p3)은 건너뛴다.
    """
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        if not VIDEO_DIR.is_dir():
            raise SystemExit(f"영상 폴더가 없습니다: {VIDEO_DIR}")
        paths = sorted(
            path
            for path in VIDEO_DIR.iterdir()
            if path.suffix.lower() in VIDEO_EXTENSIONS and not path.name.startswith(".")
        )
    selected: list[Path] = []
    skipped_test = 0
    for path in paths:
        is_test = path.stem.startswith(f"p{TEST_PERSON}_")
        if path.stem not in scenario:
            print(f"[건너뜀] {path.name}: {SCENARIO_CSV.name}에 없는 영상입니다.")
        elif is_test and not explicit:
            skipped_test += 1
        else:
            if is_test:
                print(f"[주의] {path.stem}는 최종 평가용입니다. 조정에 쓰지 마세요.")
            selected.append(path)
    if skipped_test:
        print(f"[건너뜀] 최종 평가용(p{TEST_PERSON}) 영상 {skipped_test}개 (직접 지정하면 처리)")
    if not selected:
        raise SystemExit("처리할 영상이 없습니다. data/videos/ 에 영상을 넣으세요.")
    return selected


def frames_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob(f"*{FRAMES_SUFFIX}"))


def load_frame_records(path: Path) -> list[FrameRecord]:
    with path.open(encoding="utf-8") as f:
        return [FrameRecord.model_validate_json(line) for line in f if line.strip()]


def make_events(run_dir: Path) -> list[Event]:
    """저장된 프레임 기록을 병합해 이벤트 목록을 만든다. (VLM을 부르지 않는다)"""
    events: list[Event] = []
    for path in frames_files(run_dir):
        records = load_frame_records(path)
        if not records:
            print(f"[건너뜀] {path.name}: 프레임 기록이 비어 있습니다.")
            continue
        first = records[0]
        video_start = datetime.strptime(first.video_start, START_FORMAT)
        segments = merge_records(records)
        events += to_events(first.video_id, first.room, video_start, segments, infer_gap(records))
    return sorted(events, key=lambda event: (event.start_ts, event.event_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영상 -> 프레임 기록 -> 이벤트 JSONL")
    parser.add_argument(
        "videos",
        nargs="*",
        help="처리할 영상 경로 (안 주면 data/videos/ 전부, p3 제외)",
    )
    parser.add_argument("--gap", type=float, default=FRAME_GAP_SEC, help="프레임 추출 간격(초)")
    parser.add_argument("--run", default="default", help="결과 폴더 이름 (data/outputs/<run>/)")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="VLM 없이 저장된 프레임 기록으로 이벤트만 다시 만들기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT_DIR / args.run

    if args.merge_only:
        if not frames_files(run_dir):
            raise SystemExit(f"저장된 프레임 기록이 없습니다: {run_dir}")
    else:
        scenario = load_scenario(SCENARIO_CSV)
        run_dir.mkdir(parents=True, exist_ok=True)
        for video in select_videos(args.videos, scenario):
            frames_path = run_dir / f"{video.stem}{FRAMES_SUFFIX}"
            if frames_path.exists():
                print(f"[건너뜀] {video.stem}: 이미 처리됨")
                print(f"         다시 하려면 {frames_path.name}을 지우세요")
                continue
            print(f"\n[{video.name}] 프레임 간격 {args.gap}초")
            records = describe_video(video, args.gap, scenario[video.stem])
            lines = "".join(record.model_dump_json() + "\n" for record in records)
            frames_path.write_text(lines, encoding="utf-8")

    events = make_events(run_dir)
    events_path = run_dir / "events.jsonl"
    events_path.write_text("".join(e.to_jsonl_line() + "\n" for e in events), encoding="utf-8")

    records = [r for path in frames_files(run_dir) for r in load_frame_records(path)]
    failed = sum(1 for r in records if not r.ok)
    print(f"\n프레임 {len(records)}장 (형식 오류 {failed}장) -> 이벤트 {len(events)}건")
    times = [r.elapsed_sec for r in records]
    if len(times) > 1:
        print(f"프레임당 평균 {mean(times[1:]):.1f}초 (첫 장은 모델 로딩이라 제외)")
    print(f"이벤트 저장: {events_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
