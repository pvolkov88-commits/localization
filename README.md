# UAV Visual Localization (prototype)

HTTP-сервис уточняет lat/lon БПЛА по надирному снимку и спутниковой подложке
(Esri World Imagery).

```
POST /localize
  → Georeferencer   # GSD, zoom, search center
  → MapFetcher      # XYZ tiles + disk cache
  → Localizer       # coarse template match + SIFT/ORB + RANSAC
```

Оценка по ТЗ: важнее архитектура и понимание, чем SOTA-точность.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/localize ^
  -F image=@examples/case_01_100m/query.jpg ^
  -F lat_deg=55.75 -F lon_deg=37.62 -F altitude_m=100 ^
  -F yaw_deg=15 -F pitch_deg=0 -F roll_deg=0 ^
  -F hfov_deg=60 -F search_radius_m=800
```

Docker:

```bash
docker build -t uav-visloc .
docker run --rm -p 8000:8000 -v %cd%/.cache:/app/.cache uav-visloc
```

Переменные окружения:

| Env | Default | Смысл |
|-----|---------|--------|
| `MAP_USE_STUB` | `false` | `true` — без сети, synthetic map |
| `TILE_CACHE_DIR` | `.cache/tiles` | кэш тайлов |

Генерация examples (нужен интернет):

```bash
python scripts/generate_examples.py
```

Тесты (офлайн):

```bash
pytest -q
```

## API

### `GET /health` → `{"status":"ok"}`

### `POST /localize` (multipart)

Обязательные поля: `image`, `lat_deg`, `lon_deg`, `altitude_m`, `yaw_deg`,
`pitch_deg`, `roll_deg`.

Камера: либо `(focal_mm, sensor_w_mm, sensor_h_mm)`, либо `hfov_deg`
(+ разрешение берётся из файла).

Опционально: `search_radius_m` (default 5000).

Ответ 200 — JSON с `lat_deg`, `lon_deg`, `delta_*`, `confidence`,
`processing_time_ms`, `diagnostics`.

| Код | Когда |
|-----|--------|
| 400 | битые параметры / картинка |
| 422 | matching не сошёлся / низкий confidence |
| 500 | тайлы / внутренняя ошибка |

## Map source / license

**Источник:** [Esri World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)
(`MapServer/tile/{z}/{y}/{x}`).

**Лицензия / ограничения:** данные Esri и партнёров; допускается использование
для разработки/оценки прототипа с атрибуцией. Для коммерческого/публичного
продукта нужна проверка [Esri Terms of Use](https://www.esri.com/en-us/legal/terms/full-master-agreement)
и лимитов сервиса. В ответе/README указывать: *Source: Esri World Imagery*.

Тайлы кэшируются в `.cache/tiles/esri-world-imagery/{z}/{x}/{y}.jpg`.

## Архитектура

1. **Georeferencer** — FOV/GSD из высоты и камеры; zoom Web Mercator ≈ GSD снимка;
   bbox = круг `search_radius_m` вокруг prior.
2. **MapFetcher** — XYZ-тайлы, ужимает zoom если сетка > 16×16, склеивает BGR-растр
   с углами lat/lon.
3. **Localizer**
   - поворачивает query на `-yaw` (prior north-up);
   - **coarse:** `matchTemplate` (TM_CCOEFF_NORMED) на даунскейле;
   - **fine:** SIFT (fallback ORB) + Lowe ratio + `estimateAffinePartial2D` RANSAC;
   - центр query → пиксель карты → lat/lon;
   - `confidence` из inliers / residual / coarse score; иначе 422.

## b) Ограничения метода

1. **Сезон и освещение** — листва, снег, тени; дескрипторы не совпадают с картой другого времени/часа суток.
2. **Однородная местность** — вода, поле, снег: мало уникальных точек → нестабильный RANSAC / ложный пик template match.
3. **Устаревшая подложка** — новые здания/дороги на снимке отсутствуют на Esri → outliers и провал.
4. **Сильный pitch/roll** — перспектива ≠ ортофото; текущая модель — плоская земля и affine, при pitch ≫ 0 ошибка растёт.
5. **Повтор ландшафта** — типовая застройка/поля → ambiguity, высокий score не в том месте.
6. **Большой радиус (5 км) на высоком zoom** — вынужденный downscale zoom/растра; coarse может промахнуться до fine.
7. **Чужой модальный домен** (thermal/ночь) — RGB-satellite почти бесполезен без cross-modal моделей.
8. **Ошибки высоты/yaw в meta** — неверный масштаб/поворот ухудшают overlap.

## c) Предлагаемые доработки

1. **Learned matchers** — SuperPoint + LightGlue / SuperGlue вместо SIFT.  
   [LightGlue](https://github.com/cvg/LightGlue), [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
2. **UAV↔satellite geo-localization сети** — DenseUAV / University-1652 baseline, метрическое обучение.  
   [DenseUAV](https://github.com/Dmmm1997/DenseUAV), [University-1652](https://github.com/layumi/University1652-Baseline)
3. **Place recognition + coarse-to-fine** — NetVLAD / AnyLoc на 5 км, затем локальный refine.  
   Датасет ближе к ТЗ: [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)
4. **DEM / ray casting** — учёт рельефа и pitch вместо плоской земли.
5. **Сезонные слои карты** — несколько подложек или domain randomization при обучении.
6. **Fusion с INS** — visual fix как measurement в EKF при 10 Hz INS + 1 Hz camera.
7. **Thermal pipeline** — edges / cross-modal RGB–LWIR (отдельный Localizer).

## d) Примеры работы

Сгенерировать: `python scripts/generate_examples.py`.

Симуляция по ТЗ: query croпается с той же Esri-подложки, что и search map,
с поворотом/шумом и смещённым prior.

Типичный прогон (кэш тайлов уже тёплый):

| Case | Высота | Prior error | Ошибка vs GT | conf | Время |
|------|--------|-------------|--------------|------|-------|
| `case_01_100m` | 100 м | 500 м | ~0.3 м | ~0.79 | ~1.3 s |
| `case_02_300m` | 300 м | 2 км | ~1 м | ~0.77 | ~1.5–2 s |
| `case_03_600m` | 600 м | 5 км | ~0.5 м | ~0.81 | ~1.5 s |
| `case_04_800m` | 800 м | 1.5 км + сильный photometric shift | ~0.4 м | ~0.81 | ~1.4 s |

В каждом каталоге: `query.jpg`, `meta.json`, `expected.json`, `result.json`,
`visualization.png` (query | map + стрелка prior→result).

**Разбор:** на симулированных данных (тот же сенсор/сезон, что у карты) SIFT+RANSAC
сходится стабильно, даже при 5 км prior и жёсткой фотометрии. Реальный failure mode
«другой сезон / другая карта» на такой симуляции не воспроизводится полностью —
для него нужен снимок другого времени или датасет вроде UAV-VisLoc. Смотри
`diagnostics.coarse_score`, `n_inliers`, `affine_scale` в `result.json`.

## Структура

```
app/
  main.py, models.py, pipeline.py
  georeferencer.py, map_fetcher.py, localizer.py
  tiles.py, visualize.py, exceptions.py
scripts/generate_examples.py
examples/
tests/
Dockerfile
```
