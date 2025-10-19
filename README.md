# RU ANPR — распознавание госномеров (RU/EN)

Минимальный MVP-проект для детекции и распознавания автомобильных госномеров на видео.
Поддерживает GPU (CUDA) и CPU, постобработку треков и OCR на нескольких языках.

```
.
├── models/
│   └── plate.pt
├── run.py
├── requirements.txt
├── out.csv            # создаётся после запуска
├── ref.csv            # пример/референс (не обязателен для запуска)
└── README.md
```

# Возможности
Детекция номерных знаков (модель models/plate.pt)
Трекинг объектов между кадрами
OCR с настройкой языков (--ocr_lang en,ru)
Управление частотой OCR (--ocr_every_n)
Пороговые параметры детекции/сопоставления (--conf, --iou, --match_iou)
Вывод результатов в консоль и CSV (out.csv)
Опциональный предпросмотр (--show)

# Требования
Python 3.11 
Windows / Linux / macOS
(Опционально) NVIDIA CUDA для ускорения на GPU

# Установка

1) git clone https://github.com/traadveen/ru-anpr-mvp/tree/main ru-anpr-mvp

2) cd ru-anpr-mvp

## Windows

3) python -m venv .venv

4) .venv\Scripts\activate

## Linux/macOS:

3) python3 -m venv .venv

4) source .venv/bin/activate

# Требования
5) pip install --upgrade pip
6) pip install -r requirements.txt




| Параметр        | Тип     |     По умолчанию* | Описание                                                         |
| --------------- | ------- | ----------------: | ---------------------------------------------------------------- |
| `--input`       | `str`   |                 — | Путь к видеофайлу (например, `video.mp4`).                       |
| `--model_plate` | `str`   | `models/plate.pt` | Путь к весам модели детекции номерных знаков.                    |
| `--device`      | `str`   |             `cpu` | `cuda` для GPU или `cpu`.                                        |
| `--conf`        | `float` |            `0.38` | Порог уверенности детектора (0–1).                               |
| `--iou`         | `float` |            `0.65` | IoU-порог NMS/слияния боксов.                                    |
| `--match_iou`   | `float` |            `0.40` | Минимальный IoU для сопоставления треков по кадрам.              |
| `--max_misses`  | `int`   |              `14` | Сколько кадров можно «пропустить», прежде чем трек будет закрыт. |
| `--ocr_every_n` | `int`   |               `1` | Делать OCR каждого `n`-го кадра (экономит время).                |
| `--ocr_lang`    | `str`   |           `en,ru` | Список языков OCR через запятую.                                 |
| `--show`        | флаг    |             выкл. | Показать окно с визуализацией.                                   |

# Я запускаю так
```
python run.py --input C:\ru-anpr-mvp\video.mp4 --model_plate C:\ru-anpr-mvp\models\plate.pt --device cuda --conf 0.38 --iou 0.65 --match_iou 0.40 --max_misses 14 --ocr_every_n 1 --ocr_lang en,ru --show
```


# Вывод и результаты
Во время выполнения в консоли появляются события вида:
```
[INFO] src_fps=25.00, target_fps=15.0, stride=2, device=cuda

EMIT(track 1) 00:41.84 P069XO73 conf=121.90

EMIT(track 6) 01:25.12 H444BB73 conf=286.38
```
# Практические советы

Если GPU недоступен, начни с --device cpu и уменьшай нагрузку с помощью --ocr_every_n 2 или выше.

При «дребезге» треков попробуй увеличить --match_iou и/или уменьшить --max_misses.

При большом количестве ложных срабатываний повышай --conf и (при необходимости) --iou.

Для смешанных регионов или «кривых» шрифтов попробуй порядок языков --ocr_lang ru,en.

# Типичные проблемы

Медленно? : запусти на --device cuda (если есть NVIDIA) и подними --ocr_every_n.

Окно не показывается/падает: убери --show при запуске на сервере без дисплея.

Нет файла out.csv: проверь, что видео читается (существует путь, кодек поддерживается), и что были детекции.


# Однострочный запуск
## Windows
```
python run.py --input C:\ru-anpr-mvp\video.mp4 --model_plate C:\ru-anpr-mvp\models\plate.pt --device cuda --conf 0.38 --iou 0.65 --match_iou 0.40 --max_misses 14 --ocr_every_n 1 --ocr_lang en,ru --show
```

## Linux/macOS
```
python3 run.py --input /path/to/video.mp4 --model_plate models/plate.pt --device cuda --conf 0.38 --iou 0.65 --match_iou 0.40 --max_misses 14 --ocr_every_n 1 --ocr_lang en,ru --show
```
