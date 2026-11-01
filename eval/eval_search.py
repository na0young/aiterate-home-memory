import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # retrieval import용

from retrieval.search import DATA, Embedder, EventIndex, get_time_range, load_events, parse_now, timed
NAMES = {"kure": "KURE-v1", "bge-m3": "bge-m3", "tfidf": "TF-IDF(키워드)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="kure", choices=["kure", "bge-m3", "tfidf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--qa", default=str(DATA / "qa_p1.csv"))
    ap.add_argument("--events", default=str(DATA / "gold_dev.jsonl"))
    args = ap.parse_args()

    events = load_events(args.events)
    print(f"임베딩 모델 불러오는 중: {NAMES[args.emb]}")
    (index, build_ms) = timed(EventIndex, events, Embedder(args.emb))
    with open(args.qa, encoding="utf-8-sig", newline="") as f:
        qa = list(csv.DictReader(f))

    rows, summary = [], {}
    for mode in ["시간 필터 없음", "시간 필터 있음"]:
        hits = recall = rr = 0.0
        n, ms_all = 0, []
        for q in qa:
            gold = [g for g in q["gold_event_ids"].split(";") if g]
            now = parse_now(q["now"])
            tr = get_time_range(q["question"], now) if mode == "시간 필터 있음" else None
            res, ms = timed(index.search, q["question"], k=args.k, time_range=tr)
            ms_all.append(ms)
            top = [i for i, _ in res]

            row = {"qid": q["qid"], "type": q["type"], "question": q["question"], "now": q["now"],
                   "mode": mode,
                   "time_range": f"{tr[0]:%m-%d %H:%M}~{tr[1]:%m-%d %H:%M}" if tr else "",
                   "gold": ";".join(gold), f"top{args.k}": ";".join(top), "n_found": len(top),
                   "hit": "", "recall": "", "rr": "", "ms": round(ms, 1)}
            if gold:  # 답없음 질문은 제외
                n += 1
                h = any(g in top for g in gold)
                r = sum(g in top for g in gold) / len(gold)
                first = next((i + 1 for i, t in enumerate(top) if t in gold), None)
                hits, recall, rr = hits + h, recall + r, rr + (1 / first if first else 0)
                row.update(hit=int(h), recall=round(r, 2), rr=round(1 / first, 2) if first else 0)
            rows.append(row)
        summary[mode] = (int(hits), n, recall / n, rr / n, sum(ms_all) / len(ms_all))

    # 요약
    print(f"\n임베딩 {NAMES[args.emb]} · 저장 {index.backend} · 이벤트 {len(events)}개 · "
          f"채점 질문 {summary['시간 필터 없음'][1]}개 (정답 근거가 있는 질문)")
    for mode, (h, n, r, m, ms) in summary.items():
        print(f"  {mode} | 정답 포함 {h}/{n} | Recall@{args.k} {r:.2f} | MRR {m:.2f} | 평균 검색 {ms:.0f}ms")

    # 질문별 (필터 있음)
    print(f"\n질문별 결과 (시간 필터 있음, O = 정답이 상위 {args.k}개 안에 있음)")
    for row in rows:
        if row["mode"] != "시간 필터 있음":
            continue
        mark = {1: "O", 0: "X"}.get(row["hit"], "-")
        found = row[f"top{args.k}"] or "(해당 시간에 기록 없음 → 확인 불가)"
        print(f"  {mark} {row['qid']} {row['question']}  [{row['time_range']}]  정답 {row['gold'] or '없음'}  → {found}")

    out = ROOT / "results" / f"search_{args.emb}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n질문별 결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
