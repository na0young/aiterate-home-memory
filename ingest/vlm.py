"""로컬 Ollama VLM으로 프레임 1장을 FrameResult로 변환한다."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from ollama import Client
from pydantic import ValidationError

from ingest.config import FRAME_MAX_SIDE, MAX_VLM_RETRIES, OLLAMA_HOST, VLM_MODEL
from ingest.schemas import ACTIONS, FrameResult

logger = logging.getLogger(__name__)

# gemma4:e4b(MLX)는 Ollama format(structured output)을 지원하지 않아 501이 난다.
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

_client = Client(host=OLLAMA_HOST)


class VLMResponseError(RuntimeError):
    """Ollama 응답이 FrameResult 스키마에 맞지 않을 때 발생한다."""


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
            return FrameResult.model_validate_json(_extract_json_object(_chat(messages)))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            user_prompt = f"{USER_PROMPT}\n{RETRY_PROMPT}"
            logger.warning("VLM JSON 검증 실패 (%s/%s): %s", attempt + 1, attempts, exc)
    raise VLMResponseError(
        f"Ollama 응답이 FrameResult 스키마에 맞지 않습니다. {MAX_VLM_RETRIES}회 재시도 후 실패."
    ) from last_error


def _chat(messages: list[dict]) -> str:
    """Ollama chat 호출. 같은 입력엔 같은 답이 나오게 temperature 0, thinking은 끈다."""
    response = _client.chat(
        model=VLM_MODEL,
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
