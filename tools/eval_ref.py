import csv, argparse, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -------- time helpers --------
def parse_mmss_hh(s: str) -> int:
    # 'MM:SS.hh' -> milliseconds (hh = hundredths)
    s = s.strip()
    mm, rest = s.split(":")
    ss, hh = rest.split(".")
    mm, ss, hh = int(mm), int(ss), int(hh)
    return (mm*60 + ss)*1000 + int(round(hh*10))

def fmt_ms(ms: int) -> str:
    mm = ms // 60000
    ss = (ms % 60000) // 1000
    hh = round((ms % 1000) / 10)
    if hh == 100:  # на всякий
        ss += 1; hh = 0
    return f"{mm:02d}:{ss:02d}.{hh:02d}"

# -------- plate compare (with '#') --------
ALPH = set("ABEKMHOPCTYX0123456789")
def sanitize_plate(p: str) -> str:
    p = (p or "").strip().upper()
    return "".join(ch for ch in p if (ch in ALPH or ch == "#"))

def wildcard_equal(a: str, b: str) -> bool:
    # '#' в любой строке = любой одиночный символ
    if len(a) != len(b): 
        return False
    for x, y in zip(a, b):
        if x == "#" or y == "#": 
            continue
        if x != y:
            return False
    return True

def char_accuracy(a: str, b: str) -> float:
    n = max(len(a), len(b), 1)
    a = a.ljust(n, "#"); b = b.ljust(n, "#")
    ok = 0
    for x, y in zip(a, b):
        if x == "#" or y == "#":
            continue  # не считаем неопределённые
        if x == y:
            ok += 1
    denom = sum(1 for x, y in zip(a, b) if x != "#" and y != "#")
    return (ok / denom) if denom > 0 else 0.0

# -------- data --------
@dataclass
class GT:
    start: int
    detect: int
    end: int
    plate: str
    used: bool = False

@dataclass
class Pred:
    t: int
    plate: str
    matched_gt: Optional[int] = None
    time_err_ms: Optional[int] = None
    char_acc: Optional[float] = None

def load_gt(path: str) -> List[GT]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter=';')
        header = next(rdr)
        # ожидаем: time_start;time_detect;time_end;plate_num;is_vehicle
        for r in rdr:
            if not r or len(r) < 5: 
                continue
            ts, td, te, plate, isv = r[0], r[1], r[2], r[3], r[4]
            rows.append(GT(
                start=parse_mmss_hh(ts),
                detect=parse_mmss_hh(td),
                end=parse_mmss_hh(te),
                plate=sanitize_plate(plate),
            ))
    return rows

def load_pred(path: str) -> List[Pred]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # ожидаем: time, plate_num
        for r in rdr:
            t = parse_mmss_hh(r["time"])
            p = sanitize_plate(r["plate_num"])
            rows.append(Pred(t=t, plate=p))
    return rows

# -------- matching --------
def match_predictions(gt: List[GT], pr: List[Pred]) -> Tuple[int,int,int]:
    # Жадный матчинг по времени: берём предсказание по возрастанию времени,
    # ищем первый GT, куда оно попадает по интервалу и номера «совместимы» (с '#').
    gt_sorted_idx = list(range(len(gt)))
    tp = fp = fn = 0
    for i, pred in enumerate(pr):
        matched = False
        for j in gt_sorted_idx:
            if gt[j].used:
                continue
            if not (gt[j].start <= pred.t <= gt[j].end):
                continue
            if not wildcard_equal(pred.plate, gt[j].plate):
                continue
            # match!
            gt[j].used = True
            pred.matched_gt = j
            # ошибка по времени считаем от time_detect (идеального чтения)
            pred.time_err_ms = pred.t - gt[j].detect
            pred.char_acc = char_accuracy(pred.plate, gt[j].plate)
            tp += 1
            matched = True
            break
        if not matched:
            fp += 1
    fn = sum(1 for g in gt if not g.used)
    return tp, fp, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="путь к эталонному CSV (с ';')")
    ap.add_argument("--pred", required=True, help="путь к out.csv (с ',')")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    pr = load_pred(args.pred)

    # сортируем по времени (на всякий случай)
    gt.sort(key=lambda x: x.start)
    pr.sort(key=lambda x: x.t)

    tp, fp, fn = match_predictions(gt, pr)

    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec  = tp / (tp + fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

    # агрегаты по matched
    time_err = [p.time_err_ms for p in pr if p.matched_gt is not None and p.time_err_ms is not None]
    char_accs = [p.char_acc for p in pr if p.char_acc is not None]
    mae = (sum(abs(x) for x in time_err)/len(time_err)) if time_err else 0.0

    print("\n=== SUMMARY ===")
    print(f"GT events: {len(gt)}")
    print(f"Pred events: {len(pr)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"Timing MAE vs time_detect: {mae:.1f} ms  (N={len(time_err)})")
    print(f"Char accuracy (matched): { (sum(char_accs)/len(char_accs)) if char_accs else 0.0:.3f}")

    # подробности
    print("\n--- MATCHED ---")
    for p in pr:
        if p.matched_gt is not None:
            g = gt[p.matched_gt]
            print(f"{fmt_ms(p.t)}  pred={p.plate:<10}  |  gt={g.plate:<10}  "
                  f"[{fmt_ms(g.start)}..{fmt_ms(g.end)}]  Δt={p.time_err_ms:+} ms  char_acc={p.char_acc:.2f}")
    print("\n--- FALSE POSITIVES (extra preds) ---")
    for p in pr:
        if p.matched_gt is None:
            print(f"{fmt_ms(p.t)}  pred={p.plate}")

    print("\n--- MISSED (FN) ---")
    for i, g in enumerate(gt):
        if not g.used:
            print(f"gt#{i:02d} {g.plate:<10}  [{fmt_ms(g.start)}..{fmt_ms(g.end)}], ideal@{fmt_ms(g.detect)}")

if __name__ == "__main__":
    main()
