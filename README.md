# Лабораторная работа №3. Кластеризация

Отчёт – ноутбук [`lab3.ipynb`](lab3.ipynb) (выполнен, все графики и таблицы сохранены в нём).

Датасет: [Credit Risk Benchmark Dataset](https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset) – `data/credit_risk.csv`.
Класс `dlq_2yrs` в обучении не участвует и используется только для внешних метрик.

## Состав

| Файл | Назначение |
|---|---|
| `lab3.ipynb` | отчёт: разделы 1–8 по заданию (введение, датасет, предобработка, ход работы, визуализация, заключение, источники, приложение) |
| `data/credit_risk.csv` | датасет |
| `build_notebook.py` | код ячеек, собирает `lab3.ipynb` |
| `md_text.py` | текст пояснений и выводов |

## Запуск

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab3.ipynb
```

Выполнение занимает около 2 минут.

## Результаты (16 313 заёмщиков после удаления 401 аномалии, k = 3)

| Метрика | K-means | Ward |
|---|---|---|
| Silhouette ↑ | **0.177** | 0.172 |
| Calinski–Harabasz ↑ | **3 107** | 2 551 |
| Davies–Bouldin ↓ | **1.85** | 1.91 |
| Rand index ↑ | 0.535 | **0.546** |
| Adjusted Rand ↑ | 0.071 | **0.092** |
| Jaccard ↑ | 0.316 | **0.362** |
| Устойчивость (ARI на подвыборках) ↑ | **0.98** | 0.64 |

Сегменты: «зрелые благополучные» (51%, просрочки у 38%), «молодые с ограниченной историей» (33%, 48%), «проблемные» (16%, 93%).
