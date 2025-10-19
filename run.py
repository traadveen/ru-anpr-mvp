import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
import re

# ========================
# Plate alphabet helpers
# ========================
ALPH = "ABEKMHOPCTYX"  # допустимые латинские литеры для RU номерных знаков
PLATE_RX = re.compile(rf"^[{ALPH}]\d{{3}}[{ALPH}]{{2}}\d{{2,3}}$")

# Кириллица -> латиница для визуально одинаковых букв
CYR_EQ = "\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0423\u0425"  # АВЕКМНОРСТУХ
_kir_map = {c: l for c, l in zip(CYR_EQ, ALPH)}
_kir_map.update({c.lower(): l for c, l in zip(CYR_EQ, ALPH)})
KIR2LAT = str.maketrans(_kir_map)

SIMILAR_FIX = {
    'O':'0','o':'0','I':'1','l':'1','S':'5','Z':'2'
}


def mmssms_from_ms(ts_ms: int) -> str:
    total_centis = int(round(ts_ms / 10.0))
    mm, rem_centis = divmod(total_centis, 6000)
    ss, cs = divmod(rem_centis, 100)
    return f"{mm:02d}:{ss:02d}.{cs:02d}"


def sanitize_text(t: str) -> str:
    if not t:
        return ''
    t = t.translate(KIR2LAT).upper()
    return ''.join(ch for ch in t if ch.isdigit() or ch in ALPH)


def is_valid_plate(t: str) -> bool:
    return bool(PLATE_RX.match(t))

# Позиционные приведения символов к нужной категории (буква/цифра)
DIG_FROM_LET = {'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6','T':'7','A':'4'}
LET_FROM_DIG = {'0':'O','3':'E','4':'A','6':'G','7':'T','8':'B','1':'H'}


def coerce_char_to_kind(ch: str, kind: str) -> str:
    """Жёстко приводим символ к цифре ('D') или букве ('L') набора ALPH, иначе возвращаем '#'."""
    if kind == 'D':
        if ch.isdigit():
            return ch
        return DIG_FROM_LET.get(ch, '#')
    else:  # 'L'
        if ch in ALPH:
            return ch
        return LET_FROM_DIG.get(ch, '#')


def pattern_for_len(L: int):
    # LDDDLLDD + опциональный регион D
    return (['L','D','D','D','L','L','D','D'] + (['D'] if L == 9 else []))


# ========================
# Models
# ========================

def load_models(plate_model_path: str, device: str, ocr_lang: str = 'en'):
    plate_model = YOLO(plate_model_path)
    langs = [s.strip() for s in ocr_lang.split(',') if s.strip()]
    reader = easyocr.Reader(langs or ['en'], gpu=(device == 'cuda'))
    return plate_model, reader


def detect_plate_bboxes(yolo_model: YOLO, frame_bgr, conf_thres: float, iou_thres: float, device: str):
    """Возвращает список (x1,y1,x2,y2,conf). Применяет простой фильтр геометрии для номерной таблички."""
    results = yolo_model.predict(
        frame_bgr, conf=conf_thres, iou=iou_thres, verbose=False,
        device=(0 if device == 'cuda' else 'cpu')
    )

    bboxes = []  # (x1,y1,x2,y2, conf)
    if len(results):
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), cf in zip(xyxy, conf):
                w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                ar = w / h
                if 1.6 <= ar <= 6.5 and w >= 50 and h >= 14:
                    bboxes.append((int(x1), int(y1), int(x2), int(y2), float(cf)))
    bboxes.sort(key=lambda x: x[4], reverse=True)
    return bboxes


# ========================
# OCR (с TTA и deskew)
# ========================

def _deskew_estimate(gray):
    edges = cv2.Canny(gray, 50, 120)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
    if lines is None:
        return 0.0
    angs = []
    for rho, theta in lines[:,0]:
        a = (theta*180/np.pi) - 90
        if -20 <= a <= 20:
            angs.append(a)
    return float(np.median(angs)) if angs else 0.0


def _rotate(gray, angle):
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_plate(reader: easyocr.Reader, frame_bgr: np.ndarray, bbox, max_w=420, max_h=140):
    x1, y1, x2, y2, det_conf = bbox
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return '', 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return '', 0.0

    # апскейл
    ch, cw = crop.shape[:2]
    scale = max(max_w / max(1, cw), max_h / max(1, ch))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # deskew
    angle = _deskew_estimate(gray)
    gray = _rotate(gray, -angle)

    # TTA препроцессы
    def tta_imgs(g):
        outs = [g]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        outs.append(clahe.apply(g))
        outs.append(cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,9))
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        outs.append(cv2.filter2D(g, -1, k))
        return outs

    variants = []
    base = tta_imgs(gray)
    inv = [255 - im for im in base]
    variants.extend(base + inv)

    best_text, best_conf = '', 0.0
    L_votes = None
    for img in variants:
        res = reader.readtext(img, detail=1, allowlist=ALPH + '0123456789', paragraph=False)
        if not res:
            continue
        text, conf = max(((sanitize_text(t), float(c)) for (_, t, c) in res), key=lambda x: x[1], default=('', 0.0))
        if not text:
            continue
        ok = is_valid_plate(text)
        ok_best = is_valid_plate(best_text)
        if (ok and not ok_best) or (ok == ok_best and conf > best_conf):
            best_text, best_conf = text, conf
        L = 9 if len(text) >= 9 else 8
        if L_votes is None or len(L_votes) != L:
            L_votes = [Counter() for _ in range(L)]
        norm = (text[:L]).ljust(L, '#')
        for i, ch in enumerate(norm):
            if ch != '#':
                L_votes[i][ch] += conf

    if L_votes:
        agg = ''.join((c.most_common(1)[0][0] if c else '#') for c in L_votes)
        if is_valid_plate(agg):
            return agg, max(best_conf, sum(c.most_common(1)[0][1] for c in L_votes if c)/len(L_votes))
    return best_text, best_conf


# ========================
# Tracking
# ========================
@dataclass
class Track:
    id: int
    bbox: tuple
    last_seen_frame: int
    start_ms: int
    best_plate: str = ''
    best_conf: float = 0.0
    best_ms: int = 0
    votes: list | None = None
    first_ideal_ms: int = -1
    emitted: bool = False


class SimpleTracker:
    def __init__(self, iou_match: float = 0.3, max_misses: int = 20):
        self.iou_match = iou_match
        self.max_misses = max_misses
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        area_a = max(0, (ax2-ax1)*(ay2-ay1))
        area_b = max(0, (bx2-bx1)*(by2-by1))
        union = area_a + area_b - inter + 1e-6
        return inter/union

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame_idx: int,
        ts_ms: int,
        ocr_fn,
        reader,
        frame_bgr,
        ocr_every_n: int = 2,
    ) -> List[Track]:
        assigned_det = set()

        # 1) обновляем существующие треки
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(detections):
                if j in assigned_det:
                    continue
                iou = self.iou(tr.bbox, d)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= self.iou_match:
                det = detections[best_j]
                tr.bbox = det[:4]
                det_conf = float(det[4])
                tr.last_seen_frame = frame_idx
                assigned_det.add(best_j)

                if (frame_idx % ocr_every_n) == 0:
                    text_raw, ocr_conf = ocr_fn(reader, frame_bgr, (*tr.bbox, det_conf))
                    if text_raw:
                        text = sanitize_text(text_raw)
                        L = 9 if len(text) >= 9 else 8
                        patt = pattern_for_len(L)
                        if tr.votes is None or len(tr.votes) != L:
                            tr.votes = [Counter() for _ in range(L)]
                        norm = (text[:L]).ljust(L, '#')
                        w = max(0.1, det_conf * float(ocr_conf))
                        for i, ch in enumerate(norm):
                            need = patt[i]
                            ch2 = coerce_char_to_kind(ch, need)
                            if ch2 != '#':
                                tr.votes[i][ch2] += w
                        agg = ''.join((c.most_common(1)[0][0] if c else '#') for c in tr.votes)
                        conf_sum = sum((c.most_common(1)[0][1] if c else 0.0) for c in tr.votes)
                        ideal_now = is_valid_plate(agg)
                        if ideal_now and tr.first_ideal_ms < 0:
                            tr.first_ideal_ms = ts_ms
                        better = (ideal_now and not is_valid_plate(tr.best_plate)) or (conf_sum > tr.best_conf)
                        if better:
                            tr.best_plate = agg
                            tr.best_conf = conf_sum
                            tr.best_ms = ts_ms

        # 2) создаём новые треки
        for j, d in enumerate(detections):
            if j in assigned_det:
                continue
            tid = self._next_id; self._next_id += 1
            det_conf = float(d[4])
            tr = Track(id=tid, bbox=d[:4], last_seen_frame=frame_idx, start_ms=ts_ms)
            text_raw, ocr_conf = ocr_fn(reader, frame_bgr, (*d[:4], det_conf))
            if text_raw:
                text = sanitize_text(text_raw)
                L = 9 if len(text) >= 9 else 8
                patt = pattern_for_len(L)
                tr.votes = [Counter() for _ in range(L)]
                norm = (text[:L]).ljust(L, '#')
                w = max(0.1, det_conf * float(ocr_conf))
                for i, ch in enumerate(norm):
                    need = patt[i]
                    ch2 = coerce_char_to_kind(ch, need)
                    if ch2 != '#':
                        tr.votes[i][ch2] += w
                tr.best_plate = ''.join((c.most_common(1)[0][0] if c else '#') for c in tr.votes)
                tr.best_conf = sum((c.most_common(1)[0][1] if c else 0.0) for c in tr.votes)
                tr.best_ms = ts_ms
            self.tracks[tid] = tr

        # 3) закрываем «потерянные» треки
        finished, to_delete = [], []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_seen_frame > self.max_misses:
                finished.append(tr)
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
        return finished


# ========================
# Main
# ========================

def main():
    ap = argparse.ArgumentParser(description='RU ANPR MVP (per-track best emit)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default='out.csv')
    ap.add_argument('--model_plate', required=True)
    ap.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    ap.add_argument('--target_fps', type=float, default=15.0)
    ap.add_argument('--conf', type=float, default=0.40)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--match_iou', type=float, default=0.3, help='IoU для трекера')
    ap.add_argument('--max_misses', type=int, default=24, help='через сколько пропусков закрывать трек')
    ap.add_argument('--ocr_every_n', type=int, default=2, help='делать OCR не на каждом кадре')
    ap.add_argument('--show', action='store_true', help='показывать окно с детекциями')
    ap.add_argument('--ocr_lang', type=str, default='en')
    args = ap.parse_args()

    device = args.device
    try:
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available -> falling back to CPU')
            device = 'cpu'
    except Exception:
        device = 'cpu'

    plate_model, reader = load_models(args.model_plate, device, ocr_lang=args.ocr_lang)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERR] Не удалось открыть видео: {args.input}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / max(1e-6, args.target_fps))))
    print(f"[INFO] src_fps={src_fps:.2f}, target_fps={args.target_fps}, stride={stride}, device={device}")

    out_rows = []
    tracker = SimpleTracker(iou_match=args.match_iou, max_misses=args.max_misses)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int((frame_idx / max(src_fps, 1e-6)) * 1000.0)

        if (frame_idx % stride) != 0:
            frame_idx += 1
            continue

        dets = detect_plate_bboxes(plate_model, frame, conf_thres=args.conf, iou_thres=args.iou, device=device)

        # визуализация
        if args.show:
            try:
                vis = frame.copy()
                for (x1, y1, x2, y2, cf) in dets:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{cf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                for tr in tracker.tracks.values():
                    x1, y1, x2, y2 = tr.bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)
                    if tr.best_plate:
                        cv2.putText(vis, tr.best_plate, (x1, min(vis.shape[0]-5, y2+16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2, cv2.LINE_AA)
                cv2.imshow('RU ANPR', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                pass

        finished = tracker.update(dets, frame_idx, ts_ms, ocr_plate, reader, frame, ocr_every_n=args.ocr_every_n)

        VOTE_MIN = 1.2  # минимальная сумма голосов символов
        for tr in finished:
            if tr.emitted:
                continue
            if not is_valid_plate(tr.best_plate):
                continue
            if tr.best_conf < VOTE_MIN:
                continue
            ts_emit_ms = tr.first_ideal_ms if tr.first_ideal_ms >= 0 else (tr.best_ms if tr.best_ms > 0 else tr.start_ms)
            ts_str = mmssms_from_ms(ts_emit_ms)
            out_rows.append({'time': ts_str, 'plate_num': tr.best_plate})
            tr.emitted = True
            print(f"EMIT(track {tr.id}) {ts_str} {tr.best_plate} conf={tr.best_conf:.2f}")

        frame_idx += 1

    # закрываем оставшиеся (неэмитнутые) треки одним лучшим значением
    for tr in list(tracker.tracks.values()):
        if tr.emitted:
            continue
        plate_out = tr.best_plate if tr.best_plate else '#########'
        ts_str = mmssms_from_ms(tr.best_ms if tr.best_ms > 0 else tr.start_ms)
        out_rows.append({'time': ts_str, 'plate_num': plate_out})
        print(f"EMIT(track {tr.id}, end) {ts_str} {plate_out} conf={tr.best_conf:.2f}")

    if args.show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    cap.release()

    df = pd.DataFrame(out_rows, columns=['time', 'plate_num'])
    df.to_csv(args.out, index=False)
    print(f"[OK] Записано: {args.out} ({len(df)} строк)")


if __name__ == '__main__':
    main()
