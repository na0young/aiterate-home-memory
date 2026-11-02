"""답없음 판정 평가 (검색 단계)

python eval/eval_answerable.py --emb tfidf
python eval/eval_answerable.py --emb kure      # 출력된 추천 기준값을 retrieval/search.py ANSWERABLE_RULE에 넣기

채점 기준
- 정답 근거(gold_event_ids)가 있는 질문 -> "답 있음"으로 남겨야 맞음
- 정답 근거가 없는 질문(답없음)        -> "관련 기록 없음"으로 걸러야 맞음
  (Q005처럼 근거는 있지만 답이 '확인 불가'인 질문은 근거를 찾아야 하므로 "답 있음" 쪽으로 채점)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.search import (ANSWERABLE_RULE, DATA, Embedder, EventIndex, get_time_range,
                              is_answerable, load_events, parse_now)

NAMES = {"kure": "KURE-v1", "bge-m3": "bge-m3", "tfidf": "TF-IDF(키워드)"}


def score(items, top_min, gap_min):
    """-> (답없음 정확도, 정상 질문 유지율, 전체 정확도, 맞힌 개수)"""
    rej = keep = 0
    for it in items:
        ok, _ = is_answerable(it["hits"], top_min, gap_min)
        if it["gold"] and ok:
            keep += 1
        elif not it["gold"] and not ok:
            rej += 1
    n_gold = sum(1 for it in items if it["gold"])
    n_none = len(items) - n_gold
    return rej / n_none, keep / n_gold, (rej + keep) / len(items), rej, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="kure", choices=["kure", "bge-m3", "tfidf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--qa", nargs="+", default=[str(DATA / "qa_p1.csv"), str(DATA / "qa_p2p4.csv")])
    ap.add_argument("--events", default=str(DATA / "gold_dev.jsonl"))
    args = ap.parse_args()

    events = load_events(args.events)
    print(f"임베딩 모델 불러오는 중: {NAMES[args.emb]}")
    index = EventIndex(events, Embedder(args.emb))
    qa = []
    for path in args.qa:
        with open(path, encoding="utf-8-sig", newline="") as f:
            qa += list(csv.DictReader(f))

    items = []
    for q in qa:
        tr = get_time_range(q["question"], parse_now(q["now"]))
        hits = index.search(q["question"], k=args.k, time_range=tr)
        items.append({"q": q, "gold": [g for g in q["gold_event_ids"].split(";") if g], "hits": hits})

    n_gold = sum(1 for it in items if it["gold"])
    n_none = len(items) - n_gold

    # 검색 성능 (시간 필터 있음, 근거 있는 질문만)
    hit = rr = 0
    for it in items:
        if not it["gold"]:
            continue
        top = [i for i, _ in it["hits"]]
        hit += any(g in top for g in it["gold"])
        first = next((r + 1 for r, t in enumerate(top) if t in it["gold"]), None)
        rr += 1 / first if first else 0

    # 기준값 훑기
    grid = [(round(t, 2), round(g, 2)) for t in np.arange(0, 1.0001, 0.05)
            for g in np.arange(0, 0.2001, 0.01)]
    best = max(grid, key=lambda tg: (sum(score(items, *tg)[:2]), score(items, *tg)[1]))
    rule = ANSWERABLE_RULE.get(args.emb)
    used = (rule["top_min"], rule["gap_min"]) if rule else best

    print(f"\n임베딩 {NAMES[args.emb]} · 저장 {index.backend} · 이벤트 {len(events)}개 · "
          f"질문 {len(items)}개 (근거 있음 {n_gold} / 답없음 {n_none})")
    print(f"검색 (시간 필터 있음): 정답 포함 {hit}/{n_gold} | Recall@{args.k} {hit / n_gold:.2f} | "
          f"MRR {rr / n_gold:.2f}")

    print("\n답없음 판정")
    base_keep = sum(1 for it in items if it["gold"] and it["hits"])
    base_rej = sum(1 for it in items if not it["gold"] and not it["hits"])
    print(f"  기존 (기록 0건일 때만)   | 답없음 정확도 {base_rej}/{n_none} ({base_rej / n_none:.2f}) | "
          f"정상 질문 유지 {base_keep}/{n_gold} ({base_keep / n_gold:.2f}) | "
          f"전체 {(base_rej + base_keep) / len(items):.2f}")
    a, b, c, rej, keep = score(items, *used)
    src = "search.py 설정값" if rule else "아래 추천값"
    print(f"  점수 기준 (1등≥{used[0]} 또는 차이≥{used[1]}, {src}) | 답없음 정확도 {rej}/{n_none} ({a:.2f}) | "
          f"정상 질문 유지 {keep}/{n_gold} ({b:.2f}) | 전체 {c:.2f}")
    if not rule:
        print(f"  -> 추천: ANSWERABLE_RULE['{args.emb}'] = "
              f"{{'top_min': {best[0]}, 'gap_min': {best[1]}}}")
    print("  ※ 같은 질문으로 기준값을 정해서 실제보다 좋게 나올 수 있음 -> test(p3) 질문으로 다시 확인 필요")

    print(f"\n질문별 (O = 맞게 판정)")
    rows = []
    for it in items:
        q = it["q"]
        ok, why = is_answerable(it["hits"], *used)
        correct = ok == bool(it["gold"])
        top3 = ";".join(i for i, _ in it["hits"][:3])
        print(f"  {'O' if correct else 'X'} {q['qid']} {q['question']}  정답 {';'.join(it['gold']) or '없음'}"
              f"  → {'답 있음' if ok else '확인 불가'} [{why}]  {top3}")
        rows.append({"qid": q["qid"], "type": q["type"], "question": q["question"], "now": q["now"],
                     "gold": ";".join(it["gold"]), "top3": top3,
                     "top1_score": round(it["hits"][0][1], 3) if it["hits"] else "",
                     "judged": "답 있음" if ok else "확인 불가", "reason": why, "correct": int(correct)})

    out = ROOT / "results" / f"answerable_{args.emb}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n질문별 결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
