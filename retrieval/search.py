import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

KST = timezone(timedelta(hours=9))
DATA = Path(__file__).resolve().parents[1] / "data"

# 시간 표현 -> 시간 범위
DAY_WORDS = [("그저께", -2), ("엊그제", -2), ("그제", -2), ("어제", -1), ("오늘", 0)]
PART_WORDS = [  # (단어, 시작시, 끝시)
    ("새벽", 0, 6), ("아침", 5, 11), ("오전", 0, 12), ("점심", 11, 14), ("낮", 11, 17),
    ("오후", 12, 18), ("저녁", 17, 21), ("밤", 19, 24),
]


def get_time_range(question, now):
    """'오늘 아침' -> (05:00, 11:00)
    날짜 단어 없으면 오늘, 구간 단어 없으면 하루 전체, 끝은 now까지
    """
    offset = next((d for w, d in DAY_WORDS if w in question), 0)
    day = (now + timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
    lo, hi = day, day + timedelta(days=1)
    for w, h1, h2 in PART_WORDS:
        if w in question:
            lo, hi = day + timedelta(hours=h1), day + timedelta(hours=h2)
            break
    return lo, min(hi, now)


def parse_now(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=KST)


def load_events(path=DATA / "gold_dev.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def event_text(e):
    # 임베딩용 텍스트
    objs = ", ".join(e.get("objects", []))
    return f"{e['room']} {e['action']}: {e['description']} (물건: {objs})"


class Embedder:
    def __init__(self, name="kure"):
        self.name = name
        if name == "kure":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("nlpai-lab/KURE-v1")
            try:
                self.model.encode(["테스트"])
            except Exception:  # RTX 50 + 구버전 torch면 GPU 에러 -> CPU
                print("GPU로 임베딩이 안 돼서 CPU로 바꿉니다.")
                self.model = SentenceTransformer("nlpai-lab/KURE-v1", device="cpu")
        elif name == "bge-m3":
            import ollama
            self.ollama = ollama
        elif name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
        else:
            raise ValueError("emb는 kure / bge-m3 / tfidf 중 하나")

    def encode(self, texts, fit=False):
        if self.name == "kure":
            v = self.model.encode(texts)
        elif self.name == "bge-m3":
            v = self.ollama.embed(model="bge-m3", input=texts)["embeddings"]
        else:
            v = (self.vec.fit_transform(texts) if fit else self.vec.transform(texts)).toarray()
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


# chromadb 없으면 numpy로 검색
class EventIndex:
    def __init__(self, events, embedder):
        self.emb = embedder
        self.events = {e["event_id"]: e for e in events}
        self.ids = [e["event_id"] for e in events]
        self.texts = [event_text(e) for e in events]
        self.ts = np.array([e["start_ts"] for e in events])
        self.vecs = embedder.encode(self.texts, fit=True)
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self.col = client.create_collection(f"events_{uuid.uuid4().hex[:8]}",
                                                metadata={"hnsw:space": "cosine"})
            self.col.add(ids=self.ids, embeddings=self.vecs.tolist(), documents=self.texts,
                         metadatas=[{"start_ts": int(e["start_ts"]), "room": e["room"],
                                     "action": e["action"]} for e in events])
            self.backend = "ChromaDB"
        except ImportError:
            self.col = None
            self.backend = "numpy"

    def search(self, question, k=5, time_range=None):
        """[(event_id, score), ...] 내림차순"""
        if time_range:
            lo, hi = int(time_range[0].timestamp()), int(time_range[1].timestamp())
            n_cand = int(((self.ts >= lo) & (self.ts <= hi)).sum())
        else:
            n_cand = len(self.ids)
        if n_cand == 0:
            return []
        q = self.emb.encode([question])[0]
        k = min(k, n_cand)

        if self.col is not None:
            where = ({"$and": [{"start_ts": {"$gte": lo}}, {"start_ts": {"$lte": hi}}]}
                     if time_range else None)
            res = self.col.query(query_embeddings=[q.tolist()], n_results=k, where=where)
            return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

        scores = self.vecs @ q
        if time_range:
            scores = np.where((self.ts >= lo) & (self.ts <= hi), scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


# 답없음 판정 (검색 단계)
# 기록에 없는 일을 물어도 검색은 항상 상위 k개를 돌려준다 ("오늘 손님 왔어?" -> 물 마시기 기록).
# 관련 기록이 있으면 1등 점수가 높거나 1등이 2등보다 확실히 앞선다.
# 관련 기록이 없으면 점수가 고르게 낮다 (1등과 2등 차이가 거의 없음).
# -> 1등 점수 >= top_min 이거나, 1등-2등 차이 >= gap_min 이면 "답 있음"
# 점수 범위가 임베딩마다 달라서 기준값도 임베딩별로 따로 정한다 (eval/eval_answerable.py로 측정).
ANSWERABLE_RULE = {
    "tfidf": {"top_min": 0.60, "gap_min": 0.02},
    # "kure": eval_answerable.py --emb kure 실행 후 출력된 추천값으로 채우기
}


def is_answerable(hits, top_min, gap_min):
    """hits: search() 결과 [(event_id, score), ...] -> (답 있음 여부, 이유)"""
    if not hits:
        return False, "해당 시간에 기록 없음"
    top = hits[0][1]
    gap = top - hits[1][1] if len(hits) > 1 else top
    if top >= top_min or gap >= gap_min:
        return True, f"관련 기록 있음 (1등 {top:.2f}, 2등과 차이 {gap:.2f})"
    return False, f"관련 기록 없음 (1등 {top:.2f}, 2등과 차이 {gap:.2f})"


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, (time.perf_counter() - t0) * 1000
