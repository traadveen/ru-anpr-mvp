# RU ANPR MVP (PyTorch)


## Установка
python -m venv .venv
. .venv/Scripts/activate # Windows
pip install --upgrade pip
pip install -r requirements.txt


## Модели
Положите веса детектора номерных знаков в models/plate.pt
(веса YOLO, обученные на RU прямоугольных номерах).


## Запуск
python run.py --input path/to/video.mp4 --out out.csv --model_plate models/plate.pt --device auto --target_fps 15


Параметры:
--input путь к видеофайлу
--out путь к CSV (по умолчанию out.csv)
--model_plate путь к .pt модели-детектора номеров (YOLO)
--device auto|cuda|cpu (по умолчанию auto)
--target_fps 12..15 разумно, по умолчанию 15
--conf порог детекции (по умолчанию 0.25)
--iou порог NMS (по умолчанию 0.5)
--window_ms период репортинга, мс (по умолчанию 200)


Выходной CSV:
time,plate_num
00:41.12,P069XO73
00:58.00,H800KP73