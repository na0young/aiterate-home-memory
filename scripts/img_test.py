"""이미지 1장으로 로컬 VLM 호출이 되는지 확인하고 시간을 잰다."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ingest.config import PROJECT_ROOT, TIMING_RUNS, WARMUP_RUNS
from ingest.mini_demo import describe_frame
from ingest.schemas import FrameResult


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="로컬 VLM 샘플 이미지 테스트")
    parser.add_argument("--image", required=True, help="테스트할 이미지 경로")
    args = parser.parse_args()
    image_path = Path(args.image)

    try:
        last_result, durations = _timed_runs(image_path)
    except FileNotFoundError as exc:
        print(f"오류: {exc}")
        return 1
    except ValueError as exc:
        print(f"오류: 이미지를 읽을 수 없습니다. {exc}")
        return 1

    payload = last_result.model_dump()
    stats = {
        "durations_sec": durations,
        "mean_sec": sum(durations) / len(durations),
        "min_sec": min(durations),
        "max_sec": max(durations),
    }

    print("FrameResult:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("호출별 시간(초):", [round(x, 3) for x in durations])
    print(
        f"평균 {stats['mean_sec']:.3f}s / "
        f"최소 {stats['min_sec']:.3f}s / 최대 {stats['max_sec']:.3f}s"
    )

    output_path = PROJECT_ROOT / "data" / "outputs" / "img_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"frame_result": payload, "timing": stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"저장: {output_path}")
    return 0


def _timed_runs(image_path: Path) -> tuple[FrameResult, list[float]]:
    if WARMUP_RUNS > 0:
        print(f"워밍업 {WARMUP_RUNS}회 (모델 로딩 시간 제외용)...")
    for i in range(WARMUP_RUNS):
        describe_frame(image_path)
        print(f"  warmup {i + 1}/{WARMUP_RUNS} 완료")

    durations: list[float] = []
    last = None
    for i in range(TIMING_RUNS):
        started = time.perf_counter()
        last = describe_frame(image_path)
        durations.append(time.perf_counter() - started)
        print(f"  timing {i + 1}/{TIMING_RUNS}: {durations[-1]:.3f}s")
    if last is None:
        raise RuntimeError("TIMING_RUNS가 1 이상이어야 합니다.")
    return last, durations


if __name__ == "__main__":
    raise SystemExit(main())
