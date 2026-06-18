# MHSNet: выявление мошеннических транзакций

Научно-исследовательская работа: пайплайн **Kernel PCA → LOF → Hopfield (MHSNet)** по схеме Zhao et al. (SSRN 5335578) на датасете [CaixaBank / computingvictor](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets).

---

## Пайплайн

```
Транзакции + карты + метки
        ↓
  Z-score / StandardScaler
        ↓
  Kernel PCA (нелинейные признаки)
        ↓
  LOF — локальная плотность (этап 1: выбросы)
        ↓
  Hopfield MHS — восстановление + Modern Hopfield Energy (этап 2)
        ↓
  fraud / legit
```

---

## Jupyter Notebook (основной формат для НИР)

Пошаговый отчёт с теорией, кодом, графиками и выводами:

```bash
jupyter notebook notebooks/MHSNet_Fraud_Detection_NIR.ipynb
```

Или откройте `notebooks/MHSNet_Fraud_Detection_NIR.ipynb` в Cursor / VS Code и выполняйте ячейки сверху вниз.

**Содержание notebook:** теория → загрузка данных → EDA → Kernel PCA → LOF → Hopfield → метрики → сравнение с baseline → выводы.

---

## Требования

- Python **3.10+**
- ~4 ГБ RAM (CaixaBank, 50–80K нормальных транзакций)
- Интернет для первой загрузки датасета

---

## Установка (с нуля)

```bash
# 1. Клонировать / открыть репозиторий
cd C:\my_work_repo\hopfield-network-for-fraud-nir

# 2. Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Зависимости
pip install -r requirements.txt
```

---

## Шаг 1. Скачать датасет

```bash
python scripts/download_dataset.py
```

Скрипт загрузит с Kaggle (~350 МБ) и положит файлы в `data/caixabank/`:

| Файл | Описание |
|------|----------|
| `transactions_data.csv` | ~13 млн транзакций |
| `cards_data.csv` | данные карт |
| `train_fraud_labels.json` | метки мошенничества |

> Kaggle-аккаунт не обязателен — используется `kagglehub`.

---

## Шаг 2. Запустить эксперимент

### Основной запуск (рекомендуется)

```bash
python run.py --output outputs/caixabank
```

Скрипт выполнит 6 этапов:
1. Загрузка и предобработка CaixaBank
2. Kernel PCA
3. Обучение LOF + Hopfield (MHSNet)
4. Обучение baseline-моделей
5. Оценка метрик
6. Сохранение графиков и модели

### Быстрый тест (меньше данных, ~2–3 мин)

```bash
python run.py --max-normal 20000 --eval-per-class 200 --output outputs/quick
```

---

## Все параметры `run.py`

```bash
python run.py [опции]
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--data-dir` | `data/caixabank/` | Путь к файлам датасета |
| `--output` | `outputs/` | Папка результатов |
| `--max-normal` | `80000` | Сколько нормальных транзакций загрузить |
| `--eval-per-class` | `500` | Размер val/test на класс (балансировка) |
| `--patterns` | `64` | Прототипов в сети Хопфилда |
| `--kpca-components` | `15` | Размерность после Kernel PCA |
| `--lof-neighbors` | `20` | k для LOF |
| `--pipeline-mode` | `fusion` | `fusion` или `cascade` (см. ниже) |
| `--no-kpca` | выкл. | Отключить Kernel PCA |
| `--no-lof` | выкл. | Отключить LOF |
| `--seed` | `42` | Seed |

**Примеры:**

```bash
# Режим из статьи: LOF отбирает подозрительные → Hopfield классифицирует
python run.py --pipeline-mode cascade

# Абляция: только Hopfield без KPCA и LOF
python run.py --no-kpca --no-lof

# Больше компонент KPCA
python run.py --kpca-components 20 --lof-neighbors 30
```

### Режимы pipeline

| Режим | Логика |
|-------|--------|
| `fusion` (по умолчанию) | `score = 0.35·LOF + 0.65·Hopfield` |
| `cascade` | Сначала LOF-фильтр, затем Hopfield только на подозрительных (ближе к статье) |

---

## Шаг 3. Результаты

После запуска смотрите папку `--output` (например `outputs/caixabank/`):

| Файл | Назначение |
|------|------------|
| `metrics.csv` | Таблица метрик для НИР |
| `metrics.json` | То же в JSON |
| `metrics_comparison.png` | Столбчатая диаграмма |
| `roc_curves.png` | ROC-кривые |
| `hopfield_error_distribution.png` | Распределение скоров MHSNet |
| `confusion_*.png` | Матрицы ошибок |
| `report.md` | Краткий отчёт |
| `experiment_metadata.json` | Параметры эксперимента |
| `mhsnet_detector.joblib` | **Сохранённая модель** (KPCA+LOF+Hopfield) |
| `baseline_models.joblib` | Baseline-модели |

### Метрики в консоли

```
accuracy, precision, recall, f1, roc_auc, recall_at_fpr_5
```

`recall_at_fpr_5` — Recall при FPR ≤ 5% (стандарт fraud-исследований).

---

## Шаг 4. Предсказание на новых данных

Модель ожидает **сырые признаки** (до Kernel PCA) — те же колонки, что после `caixabank_loader`.

```bash
python predict.py \
  --model-dir outputs/caixabank \
  --input data/new_features.csv \
  --output outputs/predictions.csv
```

Добавятся колонки:
- `fraud_prediction` — 0/1
- `fraud_score` — итоговый скор MHSNet
- `lof_score` — скор LOF (если включён)
- `hopfield_score` — скор Hopfield

---

## Структура проекта

```
hopfield-network-for-fraud-nir/
├── data/caixabank/              # датасет (после download_dataset.py)
├── outputs/                     # результаты экспериментов
├── scripts/download_dataset.py  # загрузка с Kaggle
├── src/
│   ├── caixabank_loader.py      # потоковая загрузка 13M строк
│   ├── feature_pipeline.py      # Kernel PCA
│   ├── mhsnet_detector.py       # LOF + Hopfield (главная модель)
│   ├── hopfield_network.py      # классическая сеть Хопфилда
│   ├── modern_hopfield_energy.py
│   ├── fraud_detector.py        # внутренний Hopfield-детектор
│   ├── baselines.py
│   ├── evaluation.py
│   └── config.py
├── run.py                       # главный скрипт
├── predict.py                   # инференс
├── docs/METHODOLOGY.md          # формулы для текста НИР
└── requirements.txt
```

---

## Настройка в коде

Файл `src/config.py`, класс `ExperimentConfig`:

```python
use_kernel_pca: bool = True
kpca_components: int = 15
kpca_kernel: str = "rbf"

use_lof: bool = True
lof_neighbors: int = 20
lof_contamination: float = 0.05

pipeline_mode: str = "fusion"   # или "cascade"
lof_weight: float = 0.35
hopfield_weight: float = 0.65
```

---

## Для текста НИР

| Раздел | Где взять |
|--------|-----------|
| Описание метода | `docs/METHODOLOGY.md` |
| Формулы KPCA, LOF, Hopfield | `docs/METHODOLOGY.md`, статья SSRN 5335578 |
| Таблица метрик | `outputs/.../metrics.csv` |
| Графики | `outputs/.../*.png` |
| Листинг программы | `src/mhsnet_detector.py` |
| Сравнение с baseline | `metrics_comparison.png` |

### Рекомендуемый порядок для отчёта

1. `python scripts/download_dataset.py`
2. `python run.py --output outputs/caixabank`
3. Скопировать `metrics.csv` в Главу 3
4. Вставить графики из `outputs/caixabank/`
5. Описать пайплайн по `docs/METHODOLOGY.md`
6. Абляция: `python run.py --no-lof` и `python run.py --no-kpca` для сравнения

---

## Устранение проблем

| Проблема | Решение |
|----------|---------|
| `CaixaBank dataset not found` | `python scripts/download_dataset.py` |
| Долго выполняется | `--max-normal 20000 --eval-per-class 200` |
| `MemoryError` | Уменьшить `--max-normal` |
| Kernel PCA медленный | `--kpca-components 10` или `--no-kpca` |
| Нет `mhsnet_detector.joblib` | Перезапустить `python run.py` |

---

## Литература

1. Zhao Y. MHSNet-SNN: Improved Outlier Detection for Fraud Detection in Financial Transactions. SSRN 5335578.
2. Ramsauer H. et al. Hopfield Networks is All You Need. ICLR 2021.
3. Breunig M. et al. LOF: Identifying Density-Based Local Outliers. SIGMOD 2000.
4. computingvictor. Financial Transactions Dataset. Kaggle 2024.
