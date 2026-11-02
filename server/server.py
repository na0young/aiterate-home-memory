"""질문 API (POST /ask)

python -m uvicorn server.server:app  ->  http://127.0.0.1:8000/docs  (루트 폴더에서 실행)

답변 모델: GEMINI_API_KEY 있으면 Gemini, 없으면 ollama gemma4:e4b
임베딩: EMB 환경변수 (기본 kure)
"""
import os
import re
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from retrieval.search import KST, Embedder, EventIndex, event_text, get_time_range, load_events, parse_now

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMB = os.getenv("EMB", "kure")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
TOP_K = 5

RULES = (
    "당신은 집 안 기록을 보고 답하는 도우미입니다. 아래 [기록]만 근거로 한국어로 짧게 답하세요.\n"
    "- 근거로 쓴 기록 번호를 [E1003]처럼 붙이세요.\n"
    "- 횟수를 묻으면 해당하는 기록 개수를 세서 답하세요.\n"
    "- 기록에 근거가 없거나, 행동이 끝까지 보이지 않아 확실하지 않으면 '확인 불가'라고 답하고 이유를 한 줄로 쓰세요.\n"
    "- 기록에 없는 내용을 추측하지 마세요."
)


app = FastAPI(title="픽셀 없는 홈 메모리", description="질문하면 기록을 근거로 답합니다.")
index = EventIndex(load_events(), Embedder(EMB))

class AskIn(BaseModel):
    # 질문 필수 입력 (앞뒤 빈칸은 지우고 1~1000자)
    question: str = Field(min_length=1, max_length=1000, examples=["오늘 아침에 엄마 약 드셨어?"])
 
    # 시각을 입력하지 않으면 None -> /ask에서 현재 시각으로 
    now: Optional[str] = Field(default=None, examples=["2026-10-27 12:30"])
 
    @field_validator("question", mode="before")
    @classmethod
    def clean_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value
 
    @field_validator("now")
    @classmethod
    def check_now(cls, value):
        if value is None or not value.strip():
            return None
 
        value = value.strip()
 
        try:
            parse_now(value)
        except ValueError:
            raise ValueError("시각은 '2026-10-27 21:00' 형식으로 입력해주세요.")
 
        return value

def generate(prompt):
    """-> (answer, model)"""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        from google import genai
        resp = genai.Client(api_key=key).models.generate_content(
            model=GEMINI_MODEL, contents=prompt)
        return resp.text.strip(), GEMINI_MODEL
    import ollama
    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0})
    return resp["message"]["content"].strip(), LLM_MODEL



@app.post("/ask")
def ask(body: AskIn):
    t0 = time.perf_counter()
    now = parse_now(body.now) if body.now else datetime.now(KST)
    lo, hi = get_time_range(body.question, now)
    hits = index.search(body.question, k=TOP_K, time_range=(lo, hi))
    found = [{"event_id": i, "score": round(s, 3), "text": event_text(index.events[i]),
              "time": index.events[i]["start_text"]} for i, s in hits]

    if not hits:
        answer, model, cited = "확인 불가 (해당 시간에 기록이 없어요)", None, []
    else:
        records = "\n".join(f"[{f['event_id']}] {f['time']} {f['text']}"
                            for f in sorted(found, key=lambda f: f["time"]))
        prompt = (f"{RULES}\n\n현재 시각: {now:%Y-%m-%d %H:%M}\n\n[기록]\n{records}\n\n"
                  f"[질문] {body.question}")
        try:
            answer, model = generate(prompt)
        except Exception as e:  # LLM 연결 실패해도 검색 결과는 반환
            answer, model = f"(답변 모델 연결 안 됨: {type(e).__name__}) 검색된 근거만 표시", None
        cited = sorted(set(re.findall(r"E\d{4}", answer)))

    return {
        "question": body.question,
        "answer": answer,
        "evidence_ids": cited,
        "unanswerable": "확인 불가" in answer,
        "time_range": [f"{lo:%Y-%m-%d %H:%M}", f"{hi:%Y-%m-%d %H:%M}"],
        "retrieved": found,
        "model": model,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "event_count": len(index.events),
    }

@app.get("/events")
def list_events():
    return list(index.events.values())