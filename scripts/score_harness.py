#!/usr/bin/env python3
"""
Score harness that reads an events CSV and computes stage scores & BSP.
CSV format (per row): t,event,id,x1,y1,x2,y2,extra...
You can append custom rows like: t, "hit", id, ..., "type=small", "label=enemy"
"""
import argparse, csv

def compute_score_stage(stage, events, duration=300.0):
    base = 0; wrong = 0
    # caps and penalties per rules
    if stage == 1:
        cap=60; big=5; small=15
    elif stage == 2:
        cap=100; big=10; small=20; wrong_pen=-30
    elif stage == 3:
        cap=140; hit=20; wrong_pen=-50
    else:
        raise ValueError("stage must be 1,2,3")

    # Read hits (you must log hits with 'event'=='hit' and include 'type=small/big' and 'label=friend/enemy')
    for e in events:
        if e["event"] != "hit": continue
        if stage in (1,2):
            typ = e.get("type","small")
            if stage==2:
                lab = e.get("label","enemy")
                if lab == "friend":
                    base += wrong_pen  # friend hit penalty
                    wrong += 1
                    continue
            base += small if typ=="small" else big
        else: # stage 3
            if e.get("correct","true") == "true":
                base += hit
            else:
                base += wrong_pen
                wrong += 1

    # Failure conditions
    if stage==2 and wrong>=2: base=0
    if stage==3 and wrong>=3: base=0

    # cap
    base = max(0, min(base, cap))

    # BSP (remaining time is last event's t subtracted from duration; or full duration if no events)
    t_last = max([float(e.get("t","0")) for e in events]+[0.0])
    remaining = max(0.0, duration - t_last)
    bsp = 20.0 * (remaining / duration)

    total = base + bsp
    return dict(base=base, bsp=bsp, total=total, wrong=wrong)

def read_log(path):
    # Accepts CSV with header, plus optional extra columns "type", "label", "correct", "t"
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--end", type=float, default=300.0, help="stage duration seconds")
    args = ap.parse_args()

    events = read_log(args.log)
    res = compute_score_stage(args.stage, events, duration=args.end)
    print(f"Stage {args.stage} Score → Base={res['base']:.1f}  BSP={res['bsp']:.1f}  Total={res['total']:.1f}  Wrong={res['wrong']}")

if __name__ == "__main__":
    main()
