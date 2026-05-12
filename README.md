# Diffract and Conquer — гиперспектральная съёмка с обычной RGB-камеры

Репозиторий — **скелет кода** под статью *«Diffract and Conquer: Hyperspectral Imaging from Any RGB Camera via Optical Encoding and Learning»* (IJCAI). Система сочетает **дифракционное кодирование** (матрица гармонических дифракционных линз 4×4, Bayer, до 48 сырых каналов) и **нейросетевую реконструкцию** спектрального куба (31 полоса, 400–700 нм, шаг 10 нм).

## Предлагаемое имя репозитория на GitHub

| Вариант | Зачем |
|--------|--------|
| **`diffract-and-conquer-hsi`** | Прямая отсылка к названию статьи, понятно для поиска. |
| **`ggpir-hsi`** | Коротко, по имени метода GGPIR (Generated Gaussian Primitives Image Restoration). |
| **`hdl-snapshot-hsi`** | Акцент на оптике (harmonic diffractive lenses) и snapshot-режиме. |

Рекомендация: **`diffract-and-conquer-hsi`** как основной публичный URL; внутри PyPI-пакет уже назван `diffract-conquer-hsi` (см. `pyproject.toml`).

## Структура репозитория

```
configs/           # гиперпараметры эксперимента и оптической модели
docs/              # архитектура и потоки данных
scripts/           # утилиты (симуляция прямой модели и т.д.)
src/diffract_conquer_hsi/
  optical/         # оптика: отбор линз, PSF, прямая модель (уравнения 3–6)
  data/            # датасеты NTIRE / ICVL / CAVE / CZ-HSDB, нормализация
  models/          # GGPIR, cmKAN++, Gaussian Primitives, спектральные блоки
  processing/      # метрики SAM / PSNR, конфиги
  training/        # train / eval (заглушки под полную реализацию)
tests/
```

## Компоненты (по статье)

### Обработка и алгоритмы (не нейросети)

- **Оптическая модель**: интегрирование сцены с PSF каждой HDL и чувствительностью Bayer \(T_c(\lambda)\) (стр. 3–4).
- **Шум**: сумма пуассоновской и гауссовской компонент (уравнение 6).
- **Выбор высот микрельефа**: дискретизация кандидатов, **жадное максимальное покрытие** спектральной сетки 31 полосы (разд. 3.1, уравнение 8).
- **Метрики**: SAM, PSNR, SSIM (разд. 4.1); нормализация по обучающей выборке послойно.

### Нейросети

- **Базовая ветка cmKAN++**: спектральный проектор и rearranger вокруг упрощённого «тела» (разд. 3.2, отсылка к cmKANlight).
- **GGPIR**: спектральный проектор с нелинейностью \(x + x \cdot \mathrm{conv2d}(x)^2\), bottleneck FFN, **слой порождённых гауссовых примитивов** (уравнение 10), расширение обратно в 31 канал.
- **Расширение до полной статьи**: независимые гиперсети на каждый выходной канал Gaussian layer, блоки Illumination Estimator / Color Transformer — см. `docs/ARCHITECTURE.md`.

## Быстрый старт

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Скрипты `training/train.py` и `scripts/simulate_forward.py` пока **заглушки**: их нужно связать с датасетами и полной реализацией Fresnel / PSF.

## Ссылка на публикацию

```bibtex
@inproceedings{pronin2026diffract,
  title     = {Diffract and Conquer: Hyperspectral Imaging from Any {RGB} Camera via Optical Encoding and Learning},
  author    = {Pronin, Alexey and Vladimirov, Daniil and Korepanov, Andrei and others},
  booktitle = {Proceedings of IJCAI},
  year      = {2026},
  note      = {Please replace with official BibTeX from proceedings}
}
```

Файл `IJCAI.pdf` в корне не включён в шаблон релиза; при публикации репозитория либо приложите ссылку на камеру-редакцию / arXiv, либо не коммитьте PDF из-за размера и прав.

## Лицензия

MIT — см. `LICENSE`. При необходимости согласуйте с авторами и грантодателем (Ministry of Economic Development of the Russian Federation, указано в статье).
