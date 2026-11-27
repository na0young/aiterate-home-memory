"""종단 평가: 이벤트 -> 시간 검색 -> 답변을 켜 둔 서버(/ask)로 한 번에 돌려서
이벤트 인식, 검색, 근거, 최종 답변, 확인 불가 판단, 응답 시간을 같이 잰다.
 
쓰는 법 (저장소 루트 폴더에서, 터미널 두 개)
1) 터미널 1: 서버를 켠다. 어떤 이벤트로 답할지 EVENTS로 고른다.
   정답 이벤트:   python3 -m uvicorn server.server:app
   VLM 이벤트:    EVENTS=data/outputs/gap1/events.jsonl python3 -m uvicorn server.server:app
   시간 필터 끔:  TIME_FILTER=off python3 -m uvicorn server.server:app
2) 터미널 2:
   python3 eval/eval_e2e.py --tag gold
   --tag 는 이번 실행의 이름. 질문별 결과는 results/e2e_<tag>.csv,
   한 줄 요약은 results/e2e_summary.csv 에 쌓여서 설정끼리 비교 표가 된다.
 
채점 기준
- 근거와 검색은 이벤트 번호가 아니라 실제 영상 구간으로 맞춘다.
  같은 영상(video_id) + 정답 이벤트와 시간이 겹침 + 같은 행동(action)이면 맞음.
  '기타 활동'은 설거지·통화·청소 등이 다 들어가는 칸이라, 정답 물건 이름(예: 화분, 청소기)이
  VLM 설명이나 물건 목록에 하나는 나와야 맞음.
  (VLM 이벤트 번호는 영상 이름으로 만들어서, 내용이 틀려도 정답 번호와 같게 나올 수 있다.)
- 이벤트 인식 Precision/Recall: 서버가 읽은 이벤트를 정답 이벤트와 1:1로 짝짓는다.
  한 동작이 여러 조각으로 쪼개지면 한 조각만 맞음, 나머지는 틀림으로 센다.
- 최종 답변:
  답없음 질문은 '확인 불가'로 답하면 맞음.
  답 있는 질문은 '확인 불가'가 아니면서, 횟수 질문은 숫자가 같고, 시간 질문은 답의 시각이
  정답과 5분 안이고, 위치 질문은 정답 방 이름이 있으면 맞음.
  그 밖의 사건 질문(했어?)은 근거가 정답 영상 구간과 행동까지 맞으면 맞음.
- 틀린 질문은 어느 단계에서 틀렸는지 적는다.
  인식: 정답 장면을 맞게 적은 이벤트가 없거나, 잘못 인식한 이벤트·쪼개진 조각을 근거로 씀 /
  검색: 맞는 이벤트는 있는데 상위 5개에 못 들어옴 /
  답변: 맞는 이벤트를 찾았는데 답이 틀림 (답없음 질문에 답해 버린 것도 여기)
- 확인 불가 정확도: 답없음 질문 중 '확인 불가'로 답한 비율과,
  답 있는 질문을 '확인 불가'로 잘못 답한 비율을 따로 본다.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, median
 
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROOMS = ["거실", "주방", "현관", "안방"]
KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
OTHER = "기타 활동"
 
 
# ---------- 서버 ----------
def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)
 
 
def ask(url, question, now):
    body = json.dumps({"question": question, "now": now}).encode("utf-8")
    req = urllib.request.Request(f"{url}/ask", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)
 
 
# ---------- 영상 구간 맞추기 ----------
def span(e):
    start = int(e["start_ts"])
    return start, int(e.get("end_ts") or start)
 
 
def same_segment(e, g, tol):
    """같은 영상이고 시간이 겹치면 True"""
    if e.get("video_id") != g.get("video_id"):
        return False
    (s1, e1), (s2, e2) = span(e), span(g)
    return s1 <= e2 + tol and e1 + tol >= s2
 
 
def matches(e, g, tol):
    """같은 영상 + 시간 겹침 + 같은 행동. '기타 활동'은 정답 물건 이름도 나와야 함"""
    if not same_segment(e, g, tol) or e.get("action") != g.get("action"):
        return False
    if g.get("action") == OTHER:
        text = e.get("description", "") + " " + " ".join(e.get("objects", []))
        return any(o in text for o in g.get("objects", []))
    return True
 
 
def overlap_sec(e, g):
    (s1, e1), (s2, e2) = span(e), span(g)
    return max(0, min(e1, e2) - max(s1, s2))
 
 
def event_pr(pred_events, gold_events, videos, tol):
    """정해진 영상들 안에서 1:1로 짝지은 이벤트 Precision/Recall"""
    pred = [e for e in pred_events if e.get("video_id") in videos]
    gold = [g for g in gold_events if g.get("video_id") in videos]
    used, tp = set(), 0
    for g in sorted(gold, key=span):
        best = None
        for i, e in enumerate(pred):
            if i in used or not matches(e, g, tol):
                continue
            if best is None or overlap_sec(e, g) > overlap_sec(pred[best], g):
                best = i
        if best is not None:
            used.add(best)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return precision, recall, tp, len(pred), len(gold)
 
 
# ---------- 답 읽기 ----------
def times_in(text):
    """답에 나온 시각들 -> 분 단위 (14:00, 오후 2시, 2시 30분, 2시 반)"""
    found = [int(h) * 60 + int(m) for h, m in re.findall(r"(\d{1,2}):(\d{2})", text)]
    for part, h, m, half in re.findall(r"(오전|오후|아침|저녁|밤|새벽)?\s*(\d{1,2})시\s*(?:(\d{1,2})분|(반))?", text):
        hour = int(h)
        if part in ("오후", "저녁", "밤") and hour < 12:
            hour += 12
        found.append(hour * 60 + (int(m) if m else 30 if half else 0))
    return found
 
 
def count_in(text):
    m = re.search(r"(\d+)\s*(?:번|회|차례)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:번|회|차례)", text)
    return KOR_NUM[m.group(1)] if m else None
 
 
def judge_answer(q, answer, unsure, evidence_recall):
    """-> (맞음 여부, 적용한 기준)"""
    gold_answer = q["gold_answer"]
    if q["answerable"].strip().upper() == "N":
        return unsure, "답없음: 확인 불가면 맞음"
    if unsure:
        return False, "답 있는데 확인 불가"
    m = re.search(r"(\d+)\s*번", gold_answer)
    if m:
        return count_in(answer) == int(m.group(1)), "횟수"
    if q["type"] == "시간":
        t = re.search(r"(\d{1,2}):(\d{2})", gold_answer)
        if t:
            gold_min = int(t.group(1)) * 60 + int(t.group(2))
            return any(abs(x - gold_min) <= 5 for x in times_in(answer)), "시간 5분 안"
    if q["type"] == "위치":
        room = next((r for r in ROOMS if r in gold_answer), None)
        if room:
            return room in answer, "위치"
    return evidence_recall > 0, "근거(구간+행동)"
 
 
def pct(x):
    return "-" if x in ("", None) else f"{x * 100:.1f}%"
 
 
def p95(values):
    s = sorted(values)
    return s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="이번 실행 이름 (예: gold, vlm_gap1, vlm_gap2, gold_notime)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--qa", nargs="+", default=[str(DATA / "qa_p1.csv"), str(DATA / "qa_p2p4.csv")])
    ap.add_argument("--gold", default=str(DATA / "gold_dev.jsonl"), help="정답 이벤트 파일")
    ap.add_argument("--split", default=None, help="dev 또는 eval만 돌리고 싶을 때")
    ap.add_argument("--tol", type=int, default=2, help="구간이 겹친다고 볼 여유 (초)")
    args = ap.parse_args()
 
    # 1. 서버 상태와 서버가 읽은 이벤트
    try:
        health = get_json(f"{args.url}/health")
        server_events = get_json(f"{args.url}/events")
    except urllib.error.URLError:
        sys.exit(f"서버({args.url})에 연결할 수 없어요. 다른 터미널에서 서버를 먼저 켜 주세요.")
    if "events_file" not in health:
        print("[주의] 서버에 EVENTS 수정이 아직 안 들어가 있어요.")
    events_file = health.get("events_file", "?")
    time_filter = health.get("time_filter", True)
    by_id = {e["event_id"]: e for e in server_events}
 
    with open(args.gold, encoding="utf-8") as f:
        gold_events = [json.loads(line) for line in f if line.strip()]
    gold_by_id = {g["event_id"]: g for g in gold_events}
    gold_videos = {g["video_id"] for g in gold_events}
 
    qa = []
    for path in args.qa:
        with open(path, encoding="utf-8-sig", newline="") as f:
            qa += [r for r in csv.DictReader(f) if not args.split or r.get("split") == args.split]
    if not qa:
        sys.exit("채점할 질문이 없어요. --qa, --split 을 확인해 주세요.")
 
    # 처리한 영상: VLM 결과 폴더에 프레임 기록(*.frames.jsonl)이 있으면 그 영상들, 없으면 정답 영상 전부
    frames = set()
    if events_file != "?":
        frames = {p.name[: -len(".frames.jsonl")] for p in (ROOT / events_file).parent.glob("*.frames.jsonl")}
    videos = (frames & gold_videos) or gold_videos
    with_events = {e.get("video_id") for e in server_events}
    print(f"이벤트 파일: {events_file} ({len(server_events)}개) · 시간 필터: {'켜짐' if time_filter else '꺼짐'}")
    if len(videos) < len(gold_videos):
        print(f"[주의] 처리한 영상이 정답 영상 {len(gold_videos)}개 중 {len(videos)}개예요. 인식 점수는 이 영상들로만 셉니다.")
    no_events = sorted(videos - with_events)
    if no_events:
        more = f" 외 {len(no_events) - 10}개" if len(no_events) > 10 else ""
        print(f"[주의] 이벤트가 하나도 없는 영상 {len(no_events)}개: {', '.join(no_events[:10])}{more}")
        print("       (영상을 아직 안 돌린 건지, 돌렸는데 사람을 못 찾은 건지 확인해 주세요)")
 
    # 2. 몸풀기 (모델을 메모리에 올리는 시간이 섞이지 않게)
    print("몸풀기 질문 1개 보내는 중 (채점 안 함)")
    ask(args.url, qa[0]["question"], qa[0]["now"])
 
    # 3. 질문마다 채점
    rows, skipped = [], []
    print(f"질문 {len(qa)}개 보내는 중\n")
    for q in qa:
        gold = [gold_by_id[g] for g in q["gold_event_ids"].split(";") if g]
        try:
            r = ask(args.url, q["question"], q["now"])
        except (urllib.error.URLError, TimeoutError) as e:  # 한 질문이 실패해도 나머지는 계속
            print(f"  ! {q['qid']} 서버 오류로 건너뜀: {e}")
            skipped.append(q["qid"])
            continue
        answer, unsure = r["answer"], bool(r["unanswerable"])
        cited = [by_id.get(i) for i in r.get("evidence_ids", [])]
        retrieved = [by_id.get(x["event_id"]) for x in r.get("retrieved", [])]
 
        ev_recall = ev_precision = seg_hit = search_recall = hit_rank = None
        recognized = True
        if gold:
            ok_cited = [c for c in cited if c and any(matches(c, g, args.tol) for g in gold)]
            ev_recall = sum(any(c and matches(c, g, args.tol) for c in cited) for g in gold) / len(gold)
            ev_precision = len(ok_cited) / len(cited) if cited else None
            seg_hit = any(c and same_segment(c, g, args.tol) for c in cited for g in gold)
            search_recall = sum(any(e and matches(e, g, args.tol) for e in retrieved) for g in gold) / len(gold)
            hit_rank = next((k + 1 for k, e in enumerate(retrieved)
                             if e and any(matches(e, g, args.tol) for g in gold)), None)
            recognized = all(any(matches(e, g, args.tol) for e in server_events) for g in gold)
        ok, rule = judge_answer(q, answer, unsure, ev_recall or 0)
        # 근거로 쓴 이벤트가 어느 정답 장면과도 안 맞거나(잘못 인식), 한 장면이 여러 조각으로 들어왔으면 인식 탓
        misread = any(c and not any(matches(c, g, args.tol) for g in gold_events) for c in cited)
        split = any(sum(bool(c) and matches(c, g, args.tol) for c in cited) > 1 for g in gold)
        if ok:
            stage = ""
        elif not recognized:
            stage = "인식"
        elif gold and search_recall < 1:
            stage = "검색"
        else:
            stage = "인식" if misread or split else "답변"
 
        def show(e, i):
            if not e:
                return f"{i}(없는 번호)"
            mark = "O" if gold and any(matches(e, g, args.tol) for g in gold) else "X"
            start, end = span(e)
            return (f"{e['event_id']} {e.get('video_id')} {e['start_text'][11:]}+{end - start}초 "
                    f"{e['action']}({e.get('description', '')[:20]}) {mark}")
 
        rows.append({
            "qid": q["qid"], "type": q["type"], "question": q["question"], "now": q["now"],
            "gold_answerable": q["answerable"], "gold_ids": q["gold_event_ids"], "gold_answer": q["gold_answer"],
            "answer": answer, "answer_ok": int(ok), "rule": rule, "fail_stage": stage, "unanswerable": int(unsure),
            "cited": " | ".join(show(e, i) for e, i in zip(cited, r.get("evidence_ids", []))),
            "evidence_recall": "" if ev_recall is None else round(ev_recall, 2),
            "evidence_precision": "" if ev_precision is None else round(ev_precision, 2),
            "segment_hit": "" if seg_hit is None else int(seg_hit),
            "search_recall5": "" if search_recall is None else round(search_recall, 2),
            "search_hit_rank": hit_rank or "",
            "retrieved": " ".join(x["event_id"] for x in r.get("retrieved", [])),
            "model": r.get("model") or "", "elapsed_sec": r["elapsed_sec"], "사람확인": "",
        })
        print(f"  {'O' if ok else 'X'} {q['qid']} {r['elapsed_sec']:6.2f}초  {q['question']}  →  {answer}"
              + (f"   [{stage}에서 틀림]" if stage else ""))
 
    # 4. 요약
    if not rows:
        sys.exit("채점된 질문이 없어요. 서버 터미널에 오류가 났는지 확인해 주세요.")
    n = len(rows)
    has_gold = [x for x in rows if x["gold_ids"]]
    yes = [x for x in rows if x["gold_answerable"].strip().upper() == "Y"]
    no = [x for x in rows if x["gold_answerable"].strip().upper() == "N"]
    precisions = [x["evidence_precision"] for x in has_gold if x["evidence_precision"] != ""]
    times = [x["elapsed_sec"] for x in rows if x["model"]]
    fails = {k: sum(x["fail_stage"] == k for x in rows) for k in ("인식", "검색", "답변")}
    ep, er, tp, n_pred, n_gold = event_pr(server_events, gold_events, videos, args.tol)
 
    s = {
        "tag": args.tag, "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "events_file": events_file, "time_filter": "on" if time_filter else "off", "questions": n,
        "event_precision": round(ep, 3), "event_recall": round(er, 3), "videos": len(videos),
        "search_recall5": round(mean(x["search_recall5"] for x in has_gold), 3) if has_gold else "",
        "search_mrr": round(mean(1 / x["search_hit_rank"] if x["search_hit_rank"] else 0 for x in has_gold), 3)
        if has_gold else "",
        "evidence_segment_hit": round(mean(x["segment_hit"] for x in has_gold), 3) if has_gold else "",
        "evidence_recall": round(mean(x["evidence_recall"] for x in has_gold), 3) if has_gold else "",
        "evidence_precision": round(mean(precisions), 3) if precisions else "",
        "answer_acc": round(mean(x["answer_ok"] for x in rows), 3),
        "answer_acc_answerable": round(mean(x["answer_ok"] for x in yes), 3) if yes else "",
        "unsure_on_unanswerable": round(mean(x["unanswerable"] for x in no), 3) if no else "",
        "false_unsure_on_answerable": round(mean(x["unanswerable"] for x in yes), 3) if yes else "",
        "fail_recognition": fails["인식"], "fail_search": fails["검색"], "fail_answer": fails["답변"],
        "avg_sec": round(mean(times), 2) if times else "", "median_sec": round(median(times), 2) if times else "",
        "p95_sec": round(p95(times), 2) if times else "",
        "model": next((x["model"] for x in rows if x["model"]), ""),
    }
 
    print(f"\n[{args.tag}] 질문 {n}개 · 이벤트 {events_file} · 시간 필터 {s['time_filter']} · 모델 {s['model'] or '없음'}")
    print(f"  이벤트 인식   Precision {pct(ep)} · Recall {pct(er)}  "
          f"(맞은 짝 {tp} / 서버 이벤트 {n_pred} / 정답 이벤트 {n_gold}, 영상 {len(videos)}개)")
    if has_gold:
        print(f"  검색          Recall@5 {pct(s['search_recall5'])} · MRR {s['search_mrr']}  (근거 있는 질문 {len(has_gold)}개)")
        print(f"  근거          정답 영상 구간과 겹침 {pct(s['evidence_segment_hit'])} · "
              f"구간+행동까지 맞음 Recall {pct(s['evidence_recall'])} / Precision {pct(s['evidence_precision'])}")
    print(f"  최종 답변     정확도 {pct(s['answer_acc'])}  (답 있는 질문만 {pct(s['answer_acc_answerable'])})")
    if no:
        print(f"  확인 불가     답없음 질문 {len(no)}개 중 확인 불가로 답함 {pct(s['unsure_on_unanswerable'])}")
    if yes:
        print(f"                답 있는 질문 {len(yes)}개 중 확인 불가로 잘못 답함 {pct(s['false_unsure_on_answerable'])}")
    print(f"  틀린 단계     인식 {fails['인식']} · 검색 {fails['검색']} · 답변 {fails['답변']}")
    if times:
        print(f"  응답 시간     평균 {s['avg_sec']}초 · 중간값 {s['median_sec']}초 · 95% {s['p95_sec']}초 "
              f"(모델을 부른 {len(times)}개)")
    if skipped:
        print(f"\n[주의] 서버 오류로 빠진 질문 {len(skipped)}개: {', '.join(skipped)} (요약 숫자에서 빠짐)")
    if any(x["answer"].startswith("(답변 모델 연결 안 됨") for x in rows):
        print("\n[주의] 답변 모델에 연결이 안 된 질문이 있어요. Ollama가 켜져 있는지 보고 다시 돌려 주세요.")
 
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    per_q = out_dir / f"e2e_{args.tag}.csv"
    with open(per_q, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = out_dir / "e2e_summary.csv"
    new = not summary.exists()
    with open(summary, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        if new:
            w.writeheader()
        w.writerow(s)
    print(f"\n질문별 결과: {per_q.relative_to(ROOT)} · 비교 표에 한 줄 추가: {summary.relative_to(ROOT)}")
 
 
if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"(걸린 시간 {time.perf_counter() - t0:.0f}초)")
 