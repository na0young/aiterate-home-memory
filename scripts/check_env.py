"""개발 환경이 VLM 호출에 필요한 조건을 갖췄는지 점검한다."""

from __future__ import annotations

import importlib
import sys

from ingest.config import OLLAMA_HOST, VLM_MODEL


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def _check_python() -> bool:
    v = sys.version_info
    ok = v[:2] >= (3, 11)
    extra = "" if ok else " (Python 3.11 이상으로 가상환경을 다시 만드세요)"
    return _report("Python 버전", ok, f"Python {v.major}.{v.minor}.{v.micro}{extra}")


def _check_imports() -> bool:
    missing = []
    for name in ("cv2", "ollama", "pydantic"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    ok = not missing
    if ok:
        detail = "cv2, ollama, pydantic import 가능"
    else:
        detail = f"{', '.join(missing)} import 실패. `pip install -r requirements.txt`를 실행하세요"
    return _report("필수 패키지", ok, detail)


def _check_server() -> bool:
    try:
        from ollama import Client

        Client(host=OLLAMA_HOST).list()
    except Exception as exc:
        detail = f"{OLLAMA_HOST} 연결 실패 ({exc}). `ollama serve`를 실행하세요"
        return _report("Ollama 서버", False, detail)
    return _report("Ollama 서버", True, f"{OLLAMA_HOST} 연결됨")


def _check_model() -> bool:
    try:
        from ollama import Client

        names = [item.model for item in Client(host=OLLAMA_HOST).list().models]
    except Exception as exc:
        return _report("VLM 모델", False, f"모델 목록을 가져오지 못했습니다 ({exc})")
    ok = VLM_MODEL in names or f"{VLM_MODEL}:latest" in names
    detail = f"{VLM_MODEL} 설치됨" if ok else f"{VLM_MODEL} 없음. `ollama pull {VLM_MODEL}`"
    return _report("VLM 모델", ok, detail)


def main() -> int:
    ok = all([_check_python(), _check_imports(), _check_server(), _check_model()])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
