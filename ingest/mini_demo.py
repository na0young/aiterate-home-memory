"""영상 -> 이벤트 변환 (ingest)

python ingest/mini_demo.py ingest p1_03_living.MOV --room 거실 --start "2026-10-27 08:10"
python ingest/mini_demo.py show
python ingest/mini_demo.py ask "오늘 약 드셨어?"
"""
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import ollama

# 설정
VLM_MODEL = "gemma4:e4b"   # RAM 부족하면 gemma4:e2b
LLM_MODEL = VLM_MODEL      # ask용
EMB_MODEL = "bge-m3"       # ask용
FRAME_GAP_SEC = 1          # 2초 -> 1초 (짧은 동작 누락)
MAX_WIDTH = 1280           # 768 -> 1280 (약통, 알약이 작게 보임)
TOP_K = 5
DB = Path("events.json")

ACTIONS = ["약 복용", "물 마시기", "식사", "택배 수령", "출입", "휴식", "기타 활동", "사람 없음"]

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "objects": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
    },
    "required": ["action", "objects", "description"],
}

VLM_PROMPT = (
    "홈캠 화면 한 장입니다. 사람이 하고 있는 행동을 아래 목록에서 하나만 고르고, "
    "눈에 띄는 물건 1~3개와 한 문장 설명을 한국어로 적으세요.\n"
    f"행동 목록: {', '.join(ACTIONS)}\n"
    "판단 기준:\n"
    "- 알약을 입에 넣거나, 약통·알약이 보이는 상태에서 컵으로 마시면 '약 복용'\n"
    "- 약통이 없고 컵으로 마시기만 하면 '물 마시기'\n"
    "- 약통을 열거나 들고만 있고 먹는 모습이 없으면 '기타 활동' (설명에 본 것만 쓰기)\n"
    "- 휴대폰을 귀에 대고 있으면 '기타 활동' (설명에 통화라고 쓰기)\n"
    "- 노트북·컴퓨터는 배경일 뿐이니, 손에 든 물건과 손의 움직임을 먼저 보세요\n"
    "규칙: 사람은 '사람'이라고만 씁니다. 성별, 나이, 얼굴 생김새, 이름, 화면이나 종이에 적힌 글자는 적지 않습니다. "
    "확실하지 않으면 '기타 활동'을 고릅니다. JSON으로만 답합니다."
)

ANSWER_RULES = (
    "당신은 집 안 기록을 보고 답하는 도우미입니다. 아래 [기록]만 근거로 한국어로 짧게 답하세요. "
    "근거로 쓴 기록 번호를 [E번호] 형태로 붙이세요. "
    "기록에 근거가 없으면 '기록에서 확인되지 않습니다'라고만 답하고 추측하지 마세요."
)


def load_events():
    return json.loads(DB.read_text(encoding="utf-8")) if DB.exists() else []


def save_events(events):
    DB.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


def describe_frame(jpg_bytes):
    """프레임 1장 -> 이벤트 dict"""
    resp = ollama.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": VLM_PROMPT, "images": [jpg_bytes]}],
        format=EVENT_SCHEMA,
        options={"temperature": 0},
    )
    return json.loads(resp["message"]["content"])


def ingest(video, room, start):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video} (파일 이름을 영문으로 바꿔 보세요)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * FRAME_GAP_SEC)))

    frames, times = [], []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame)
        t0 = time.time()
        try:
            ev = describe_frame(buf.tobytes())
        except (json.JSONDecodeError, KeyError):
            ev = {"action": "기타 활동", "objects": [], "description": "(형식 오류로 해석 실패)"}
        times.append(time.time() - t0)
        ev["time"] = start + timedelta(seconds=idx / fps)
        frames.append(ev)
        print(f"  {ev['time']:%H:%M:%S}  {ev['action']:<6} {ev['description']}  ({times[-1]:.1f}초)")
    cap.release()

    # 같은 행동이 이어지면 병합, 사람 없음은 제외
    # TODO: 짧게 끊긴 구간 흡수, 설명은 최빈값으로
    merged = []
    for ev in frames:
        if ev["action"] == "사람 없음":
            merged.append(None)
            continue
        last = merged[-1] if merged else None
        if last and last["action"] == ev["action"]:
            last["end"] = ev["time"]
            last["objects"] = sorted(set(last["objects"]) | set(ev["objects"]))
        else:
            merged.append({"action": ev["action"], "objects": sorted(set(ev["objects"])),
                           "description": ev["description"], "start": ev["time"], "end": ev["time"]})
    new = [m for m in merged if m]

    events = load_events()
    for m in new:
        events.append({
            "id": f"E{len(events) + 1}", "room": room, "action": m["action"],
            "objects": m["objects"], "description": m["description"],
            "start": m["start"].strftime("%Y-%m-%d %H:%M:%S"),
            "end": m["end"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    save_events(events)

    print(f"\n프레임 {len(frames)}장 -> 이벤트 {len(new)}건 저장 (누적 {len(events)}건, {DB})")
    if times:
        print(f"프레임당 평균 {sum(times) / len(times):.1f}초")
    if len(times) > 1:
        print(f"첫 장 빼고 평균 {sum(times[1:]) / len(times[1:]):.1f}초")


def event_line(e):
    objs = ", ".join(e["objects"]) or "-"
    return f"[{e['id']}] {e['start']}~{e['end'][11:]} {e['room']} | {e['action']} | {e['description']} | 물건: {objs}"


def show():
    events = load_events()
    print("\n".join(event_line(e) for e in events) or "저장된 이벤트가 없습니다.")


def ask(question, now):
    events = load_events()
    if not events:
        raise SystemExit("저장된 이벤트가 없습니다. 먼저 ingest를 실행하세요.")
    t0 = time.time()
    texts = [event_line(e) for e in events]
    vecs = np.array(ollama.embed(model=EMB_MODEL, input=[question] + texts)["embeddings"])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = vecs[1:] @ vecs[0]
    top = np.argsort(-scores)[:TOP_K]

    records = "\n".join(texts[i] for i in sorted(top))
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_RULES},
            {"role": "user", "content": f"현재 시각: {now:%Y-%m-%d %H:%M}\n\n[기록]\n{records}\n\n[질문] {question}"},
        ],
        options={"temperature": 0},
    )
    print(f"\n질문: {question}\n답변: {resp['message']['content'].strip()}\n")
    print("검색된 근거 (유사도 순)")
    for i in top:
        print(f"  {scores[i]:.2f}  {texts[i]}")
    print(f"\n응답 시간 {time.time() - t0:.1f}초")


def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="픽셀 없는 홈 메모리 미니 데모")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest", help="영상을 이벤트로 바꿔 저장")
    a.add_argument("video")
    a.add_argument("--room", default="거실")
    a.add_argument("--start", type=parse_time, default=datetime.now().replace(second=0, microsecond=0),
                   help='영상이 시작된 시각, 예: "2026-10-11 08:00"')
    sub.add_parser("show", help="저장된 이벤트 보기")
    q = sub.add_parser("ask", help="질문하기")
    q.add_argument("question")
    q.add_argument("--now", type=parse_time, default=datetime.now(), help="'오늘'의 기준 시각")
    args = p.parse_args()

    if args.cmd == "ingest":
        ingest(Path(args.video), args.room, args.start)
    elif args.cmd == "show":
        show()
    else:
        ask(args.question, args.now)
