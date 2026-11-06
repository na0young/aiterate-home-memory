"""환경변수와 실행 상수를 한곳에서 읽는다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL: str = os.getenv("VLM_MODEL", "gemma4:e4b")
FRAME_MAX_SIDE: int = int(os.getenv("FRAME_MAX_SIDE", "1280"))
FRAME_GAP_SEC: float = float(os.getenv("FRAME_GAP_SEC", "1"))
LLM_MODEL: str = os.getenv("LLM_MODEL", VLM_MODEL)  # ask 답변용
EMB_MODEL: str = os.getenv("EMB_MODEL", "bge-m3")  # ask 검색용
EVENTS_DB: Path = PROJECT_ROOT / "events.json"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = DATA_DIR / "outputs"
VIDEO_DIR: Path = DATA_DIR / "videos"
SCENARIO_CSV: Path = DATA_DIR / "scenario.csv"
TEST_PERSON: int = 3  # 최종 평가용 영상(p3). 개발 중에는 처리하지 않는다.
WARMUP_RUNS: int = int(os.getenv("WARMUP_RUNS", "1"))
TIMING_RUNS: int = int(os.getenv("TIMING_RUNS", "3"))
MAX_VLM_RETRIES: int = 2
