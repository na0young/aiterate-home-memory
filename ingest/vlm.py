"""로컬 Ollama VLM으로 프레임 1장을 FrameResult로 바꾼다."""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from ollama import Client

from ingest.config import FRAME_MAX_SIDE, MAX_VLM_RETRIES, OLLAMA_HOST, VLM_MODEL
from ingest.schemas import ACTIONS, FrameResult

logger = logging.getLogger(__name__)

ABSENT = "사람 없음"  # 사람이 안 보이면 행동 대신 이 값을 고르게 한다
PROMPT_ACTIONS = [*ACTIONS, ABSENT]

# Try1 - 미니데모와 동일 프롬프트
# VLM_PROMPT = (
#     "홈캠 화면 한 장입니다. 사람이 하고 있는 행동을 아래 목록에서 하나만 고르고, "
#     "눈에 띄는 물건 1~3개와 한 문장 설명을 한국어로 적으세요.\n"
#     f"행동 목록: {', '.join(PROMPT_ACTIONS)}\n"
#     "판단 기준:\n"
#     "- 알약을 입에 넣거나, 약통·알약이 보이는 상태에서 컵으로 마시면 '약 복용'\n"
#     "- 약통이 없고 컵으로 마시기만 하면 '물 마시기'\n"
#     "- 약통을 열거나 들고만 있고 먹는 모습이 없으면 '기타 활동' (설명에 본 것만 쓰기)\n"
#     "- 휴대폰을 귀에 대고 있으면 '기타 활동' (설명에 통화라고 쓰기)\n"
#     "- 노트북·컴퓨터는 배경일 뿐이니, 손에 든 물건과 손의 움직임을 먼저 보세요\n"
#     "규칙: 사람은 '사람'이라고만 씁니다. 성별, 나이, 얼굴 생김새, 이름, "
#     "화면이나 종이에 적힌 글자는 적지 않습니다. "
#     "확실하지 않으면 '기타 활동'을 고릅니다. JSON으로만 답합니다."
# )

# Try2 - 일반화된 프롬프트
VLM_PROMPT = (
    "홈캠 화면 한 장입니다. 사람이 하고 있는 행동을 아래 목록에서 하나만 고르고, "
    "눈에 띄는 물건 1~3개와 한 문장 설명을 한국어로 적으세요.\n"
    f"행동 목록: {', '.join(PROMPT_ACTIONS)}\n"
    "행동 정의:\n"
    "- 약 복용: 알약을 입에 넣거나, 약통·알약이 보이는 상태에서 컵으로 마신다.\n"
    "- 물 마시기: 컵이나 병으로 물이나 음료를 마신다. 약통은 보이지 않는다.\n"
    "- 식사: 식탁에 앉아 음식을 먹는다.\n"
    "- 택배 수령: 문을 열고 택배 상자를 받거나 들여온다.\n"
    "- 출입: 신발을 신고 외출하거나, 장바구니 등을 들고 귀가한다.\n"
    "- 기상: 잠자리에서 일어난다.\n"
    "- 휴식: 앉거나 누워서 TV나 휴대폰을 보거나 낮잠을 잔다.\n"
    "- 기타 활동: 위에 해당하지 않는 모든 행동. 무엇을 하는지 설명에 쓴다.\n"
    "- 사람 없음: 사람이 보이지 않는다.\n"
    "판단 기준:\n"
    "- 약통을 열거나 들고만 있고 먹는 모습이 없으면 '약 복용'이 아니라 '기타 활동'입니다. "
    "(설명에 본 것만 쓰기)\n"
    "- 손의 움직임과 손에 든 물건을 먼저 보세요. "
    "노트북·컴퓨터 등 배경에 있는 물건은 근거가 아닙니다.\n"
    "규칙: 사람은 '사람'이라고만 씁니다. 성별, 나이, 얼굴 생김새, 이름, "
    "화면이나 종이에 적힌 글자는 적지 않습니다. "
    "확실하지 않으면 '기타 활동'을 고릅니다. JSON으로만 답합니다."
)


KEY_HINT = (
    "JSON 키는 action, objects(리스트), description 세 개만 쓴다. 코드블록 없이 JSON만 출력한다."
)
RETRY_PROMPT = (
    f"이전 답은 형식에 맞지 않았다. action은 {PROMPT_ACTIONS} 중 하나만 쓰고, "
    "JSON 키는 action, objects, description 세 개만 쓴다."
)

# 모델이 주지 않는 값이다. 이웃 프레임 일치도로 계산하기 전까지 자리표시로 쓴다.
PLACEHOLDER_CONFIDENCE = 1.0

_client = Client(host=OLLAMA_HOST)


class VLMResponseError(RuntimeError):
    """Ollama 응답이 끝내 형식에 맞지 않을 때 발생한다."""

    def __init__(self, message: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


def describe_frame(frame: np.ndarray) -> tuple[FrameResult, int]:
    """프레임(OpenCV BGR 배열) 1장을 VLM에 넣어 (결과, 시도 횟수)를 돌려준다.

    시도 횟수가 1이면 재시도 없이 성공한 것이다.
    """
    jpeg = _to_jpeg(frame)
    prompt = VLM_PROMPT
    attempts = MAX_VLM_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        messages = [
            {"role": "system", "content": KEY_HINT},
            {"role": "user", "content": prompt, "images": [jpeg]},
        ]
        try:
            return _parse(_chat(messages)), attempt + 1
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            prompt = f"{VLM_PROMPT}\n{RETRY_PROMPT}"
            logger.warning("VLM 응답 형식 오류 (%s/%s): %s", attempt + 1, attempts, exc)
    raise VLMResponseError(
        f"VLM 응답이 형식에 맞지 않습니다. {MAX_VLM_RETRIES}회 재시도 후 실패.",
        attempts=attempts,
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


def _parse(text: str) -> FrameResult:
    """응답에서 첫 '{'부터 마지막 '}'까지를 JSON으로 읽어 FrameResult로 바꾼다."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"JSON 객체를 찾지 못했습니다: {text[:200]!r}")
    data = json.loads(text[start : end + 1])
    action = data["action"]
    if action not in PROMPT_ACTIONS:
        raise ValueError(f"행동 목록에 없는 값입니다: {action!r}")
    present = action != ABSENT
    return FrameResult(
        person_present=present,
        action=action if present else "기타 활동",
        objects=data["objects"],
        description=data["description"],
        confidence=PLACEHOLDER_CONFIDENCE,
        observed=True,
    )


def _to_jpeg(frame: np.ndarray) -> bytes:
    """프레임을 긴 변이 FRAME_MAX_SIDE 이하가 되게 줄여서 JPEG 바이트로 바꾼다."""
    height, width = frame.shape[:2]
    scale = FRAME_MAX_SIDE / max(height, width)
    if scale < 1:
        size = (int(width * scale), int(height * scale))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG 인코딩에 실패했습니다.")
    return buffer.tobytes()
