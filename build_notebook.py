"""Сборка ноутбука lab3.ipynb из ячеек.

Запуск:
    .venv/bin/python build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace lab3.ipynb
"""
import nbformat as nbf
from pathlib import Path

from md_text import MD

ROOT = Path(__file__).parent
cells = []


def md(key):
    cells.append(nbf.v4.new_markdown_cell(MD[key].strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =================================================================== 1. Введение
md("title")
md("intro")

# =================================================================== 2. Датасет
md("dataset")
code(r'''
%matplotlib inline
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score,
    rand_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)
from sklearn.metrics.cluster import pair_confusion_matrix

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

FEATURE_RU = {
    "rev_util":    "Использование кредитного лимита",
    "age":         "Возраст",
    "late_30_59":  "Просрочки 30–59 дн.",
    "debt_ratio":  "Долговая нагрузка",
    "monthly_inc": "Месячный доход",
    "open_credit": "Открытые кредиты",
    "late_90":     "Просрочки 90+ дн.",
    "real_estate": "Ипотечные кредиты",
    "late_60_89":  "Просрочки 60–89 дн.",
    "dependents":  "Иждивенцы",
    "dlq_2yrs":    "Просрочка за 2 года (класс)",
}
''')

code(r'''
df_raw = pd.read_csv("data/credit_risk.csv")
TARGET = "dlq_2yrs"
FEATURES = [c for c in df_raw.columns if c != TARGET]

print("Размерность:", df_raw.shape)
print("Пропусков всего:", int(df_raw.isna().sum().sum()))
print("Полных дубликатов:", int(df_raw.duplicated().sum()))
info = pd.DataFrame({
    "описание": [FEATURE_RU[c] for c in df_raw.columns],
    "тип": df_raw.dtypes.astype(str).values,
    "уникальных": df_raw.nunique().values,
    "пропусков": df_raw.isna().sum().values,
    "доля нулей": (df_raw == 0).mean().round(3).values,
}, index=df_raw.columns)
display(info)
display(df_raw.head())
''')

code(r'''
desc = df_raw[FEATURES].describe(percentiles=[.01, .25, .5, .75, .99]).T
desc["асимметрия"] = df_raw[FEATURES].skew()
desc["эксцесс"] = df_raw[FEATURES].kurt()
display(desc)
print("Баланс классов:")
display(df_raw[TARGET].value_counts().rename("кол-во").to_frame().assign(доля=lambda t: t["кол-во"] / len(df_raw)))
''')
md("dataset_desc")

md("hist_raw")
code(r'''
def plot_hists(data, title, bins=50):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(data[col], bins=bins, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"{col}\n{FEATURE_RU[col]}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

plot_hists(df_raw, "Распределения признаков (исходные данные)")

# Проверка нормальности: тест Д'Агостино–Пирсона (n велико) и Шапиро–Уилка на подвыборке 500
rng = np.random.default_rng(SEED)
sub = df_raw[FEATURES].sample(500, random_state=SEED)
norm_tab = pd.DataFrame({
    "асимметрия": df_raw[FEATURES].skew(),
    "эксцесс": df_raw[FEATURES].kurt(),
    "p (Д'Агостино)": [stats.normaltest(df_raw[c]).pvalue for c in FEATURES],
    "W (Шапиро, n=500)": [stats.shapiro(sub[c]).statistic for c in FEATURES],
    "p (Шапиро, n=500)": [stats.shapiro(sub[c]).pvalue for c in FEATURES],
})
norm_tab["нормальное?"] = np.where(norm_tab["p (Шапиро, n=500)"] > 0.05, "да", "нет")
display(norm_tab)
''')
md("hist_raw_desc")

md("outliers_raw")
code(r'''
fig, axes = plt.subplots(2, 5, figsize=(20, 6.5))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.boxplot(x=df_raw[col], ax=ax, color="lightcoral", fliersize=2)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
fig.suptitle("Диаграммы размаха (исходные данные)", fontsize=14)
plt.tight_layout()
plt.show()

def iqr_outliers(s, k=1.5):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return ((s < q1 - k * iqr) | (s > q3 + k * iqr))

out_tab = pd.DataFrame({
    "выбросов IQR (1.5)": [int(iqr_outliers(df_raw[c]).sum()) for c in FEATURES],
    "выбросов |z|>3": [int((np.abs(stats.zscore(df_raw[c])) > 3).sum()) for c in FEATURES],
    "максимум": df_raw[FEATURES].max().values,
    "99-й перцентиль": df_raw[FEATURES].quantile(.99).values,
}, index=FEATURES)
out_tab["доля IQR, %"] = (out_tab["выбросов IQR (1.5)"] / len(df_raw) * 100).round(2)
display(out_tab)

late_cols = ["late_30_59", "late_60_89", "late_90"]
anom = pd.DataFrame({
    "служебные коды 96/98 в просрочках": (df_raw[late_cols] >= 96).any(axis=1),
    "доход < 100 (0 или 1)": df_raw["monthly_inc"] < 100,
    "rev_util > 2": df_raw["rev_util"] > 2,
})
anom["любая аномалия"] = anom.any(axis=1)
display(anom.sum().rename("строк").to_frame())
print("Медиана debt_ratio при доходе < 100:", df_raw.loc[anom["доход < 100 (0 или 1)"], "debt_ratio"].median(),
      "| при доходе >= 100:", round(df_raw.loc[~anom["доход < 100 (0 или 1)"], "debt_ratio"].median(), 3))
''')
md("outliers_raw_desc")

md("informativeness")
code(r'''
corr = df_raw.corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title("Корреляции Спирмена (включая класс dlq_2yrs)")
plt.tight_layout()
plt.show()

mi = mutual_info_classif(df_raw[FEATURES], df_raw[TARGET], random_state=SEED)
inf_tab = pd.DataFrame({
    "коэф. вариации": (df_raw[FEATURES].std() / df_raw[FEATURES].mean()).values,
    "уникальных": df_raw[FEATURES].nunique().values,
    "|ρ| с классом": corr.loc[FEATURES, TARGET].abs().values,
    "взаимная информация с классом": mi,
    "макс. |ρ| с другим признаком": [corr.loc[c, [f for f in FEATURES if f != c]].abs().max() for c in FEATURES],
}, index=FEATURES).sort_values("взаимная информация с классом", ascending=False)
display(inf_tab)

fig, ax = plt.subplots(figsize=(9, 4))
inf_tab["взаимная информация с классом"].plot.barh(ax=ax, color="teal")
ax.invert_yaxis()
ax.set_title("Информационная значимость признаков (взаимная информация с dlq_2yrs)")
plt.tight_layout()
plt.show()
''')
md("informativeness_desc")

md("conditions")

# =================================================================== 3. Предобработка
md("prep")
code(r'''
# 1) удаление аномальных записей (ошибки данных, а не «редкие клиенты»)
df = df_raw.loc[~anom["любая аномалия"]].reset_index(drop=True)
y = df[TARGET].values                       # класс: НЕ используется при обучении
X_raw = df[FEATURES].copy()
print(f"Удалено аномальных строк: {len(df_raw) - len(df)} ({(len(df_raw) - len(df)) / len(df_raw):.2%}); осталось {len(df)}")

# 2) винзоризация хвостов по 1-му и 99-му перцентилям
lo, hi = X_raw.quantile(.01), X_raw.quantile(.99)
X_w = X_raw.clip(lower=lo, upper=hi, axis=1)

# 3) логарифмирование признаков с правосторонней асимметрией
LOG_COLS = ["rev_util", "debt_ratio", "monthly_inc", "open_credit", "real_estate",
            "dependents", "late_30_59", "late_60_89", "late_90"]
X_t = X_w.copy()
X_t[LOG_COLS] = np.log1p(X_t[LOG_COLS])

# 4) стандартизация
scaler = StandardScaler()
Z = scaler.fit_transform(X_t)
Zdf = pd.DataFrame(Z, columns=FEATURES)

skew_cmp = pd.DataFrame({
    "асимметрия (после удаления аномалий)": X_raw.skew(),
    "после винзоризации": X_w.skew(),
    "после log1p": X_t.skew(),
    "среднее после Z": Zdf.mean().round(6).values,
    "СКО после Z": Zdf.std(ddof=0).round(6).values,
    "|z|>3 после обработки": (np.abs(Zdf) > 3).sum().values,
})
display(skew_cmp)
''')
code(r'''
plot_hists(X_t, "Распределения после удаления аномалий, винзоризации и log1p", bins=40)

fig, ax = plt.subplots(figsize=(16, 5))
sns.boxplot(data=Zdf, ax=ax, palette="Set2", fliersize=1.5)
ax.set_title("Стандартизованные признаки (z-оценки)")
ax.set_ylabel("z")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
''')
md("prep_desc")

md("scaler_cmp")
code(r'''
scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
rows = []
for name, sc in scalers.items():
    A = sc.fit_transform(X_t)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(A)
    spread = A.std(axis=0)
    rows.append({
        "масштабирование": name,
        "СКО признаков: мин": spread.min(), "СКО признаков: макс": spread.max(),
        "silhouette (k=3)": silhouette_score(A, km.labels_, sample_size=6000, random_state=SEED),
        "Calinski–Harabasz": calinski_harabasz_score(A, km.labels_),
        "Davies–Bouldin": davies_bouldin_score(A, km.labels_),
        "размеры кластеров": np.bincount(km.labels_).tolist(),
    })
display(pd.DataFrame(rows).set_index("масштабирование"))
''')
md("scaler_desc")

md("tendency")
code(r'''
def hopkins(A, m=None, seed=SEED):
    """Статистика Хопкинса: ~0.5 – равномерные данные, -> 1 – выраженная кластерная структура."""
    rng = np.random.default_rng(seed)
    n, d = A.shape
    m = m or int(0.05 * n)
    nn = NearestNeighbors(n_neighbors=2).fit(A)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(A[idx], n_neighbors=2)[0][:, 1]                   # расстояние до ближайшего реального соседа
    U = rng.uniform(A.min(axis=0), A.max(axis=0), size=(m, d))
    u = nn.kneighbors(U, n_neighbors=1)[0][:, 0]                         # от случайной точки до данных
    return u.sum() / (u.sum() + w.sum())

H = [hopkins(Z, seed=s) for s in range(10)]
print(f"Статистика Хопкинса (10 запусков): {np.mean(H):.3f} ± {np.std(H):.3f}")

pca_full = PCA().fit(Z)
evr = pca_full.explained_variance_ratio_
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar(range(1, 11), evr, color="steelblue", label="доля дисперсии")
axes[0].plot(range(1, 11), np.cumsum(evr), "o-", color="darkred", label="накопленная")
axes[0].set_xlabel("главная компонента"); axes[0].set_title("PCA стандартизованных данных"); axes[0].legend()
P2 = pca_full.transform(Z)[:, :2]
hb = axes[1].hexbin(P2[:, 0], P2[:, 1], gridsize=60, bins="log", cmap="viridis")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("Плотность объектов в плоскости PC1–PC2")
plt.colorbar(hb, ax=axes[1], label="log(кол-во)")
plt.tight_layout()
plt.show()
print("Доля дисперсии PC1+PC2:", round(evr[:2].sum(), 3), "| PC1..PC4:", round(evr[:4].sum(), 3))
loadings = pd.DataFrame(pca_full.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
display(loadings.round(3))
''')
md("tendency_desc")

md("pairplot")
code(r'''
pp_sample = Zdf.sample(2000, random_state=SEED)
g = sns.pairplot(pp_sample, corner=True, plot_kws=dict(s=6, alpha=0.35, edgecolor="none"),
                 diag_kws=dict(bins=30, color="steelblue"), height=1.6)
g.fig.suptitle("Матрица диаграмм рассеивания (стандартизованные признаки, 2000 объектов)", y=1.01, fontsize=14)
plt.show()
''')
md("pairplot_desc")

md("methods_choice")

# =================================================================== 4. Ход работы
md("k_selection")
code(r'''
K_RANGE = range(2, 11)
km_rows = []
km_models = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    km_models[k] = km
    km_rows.append({
        "k": k, "WCSS (inertia)": km.inertia_,
        "silhouette": silhouette_score(Z, km.labels_),
        "Calinski–Harabasz": calinski_harabasz_score(Z, km.labels_),
        "Davies–Bouldin": davies_bouldin_score(Z, km.labels_),
        "мин. размер кластера": np.bincount(km.labels_).min(),
    })
km_sel = pd.DataFrame(km_rows).set_index("k")

# «локоть» формально: точка максимального отклонения кривой WCSS от хорды (метод Kneedle)
w = km_sel["WCSS (inertia)"].values
ks = np.array(list(K_RANGE))
xn = (ks - ks.min()) / (ks.max() - ks.min())
yn = (w - w.min()) / (w.max() - w.min())
elbow_k = int(ks[np.argmax((1 - xn) - yn)])
km_sel["снижение WCSS, %"] = (-km_sel["WCSS (inertia)"].pct_change() * 100).round(1)
display(km_sel)
print("Локоть (Kneedle):", elbow_k,
      "| максимум silhouette:", km_sel["silhouette"].idxmax(),
      "| максимум CH:", km_sel["Calinski–Harabasz"].idxmax(),
      "| минимум DB:", km_sel["Davies–Bouldin"].idxmin())
''')
code(r'''
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plots = [("WCSS (inertia)", "Метод локтя: WCSS(k)", "steelblue"),
         ("silhouette", "Средний коэффициент силуэта (↑)", "seagreen"),
         ("Calinski–Harabasz", "Индекс Калинского–Харабаза (↑)", "darkorange"),
         ("Davies–Bouldin", "Индекс Дэвиса–Болдина (↓)", "firebrick")]
for ax, (col, title, color) in zip(axes.ravel(), plots):
    ax.plot(km_sel.index, km_sel[col], "o-", color=color)
    ax.set_title(title); ax.set_xlabel("k"); ax.set_xticks(list(K_RANGE))
axes[0, 0].axvline(elbow_k, ls="--", color="grey"); axes[0, 0].text(elbow_k + 0.1, w.max() * 0.95, f"локоть k={elbow_k}")
axes[0, 1].axvline(km_sel["silhouette"].idxmax(), ls="--", color="grey")
fig.suptitle("Подбор числа кластеров для K-means", fontsize=14)
plt.tight_layout()
plt.show()
''')
code(r'''
def silhouette_plot(ax, A, labels, title):
    s = silhouette_samples(A, labels)
    y_low = 10
    palette = sns.color_palette("tab10")
    for c in np.unique(labels):
        sc = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(y_low, y_low + len(sc)), 0, sc, color=palette[c % 10], alpha=0.8)
        ax.text(-0.08, y_low + len(sc) / 2, str(c), fontsize=9)
        y_low += len(sc) + 60
    ax.axvline(s.mean(), color="red", ls="--")
    ax.set_title(f"{title}\nсредний силуэт = {s.mean():.3f}, доля s<0 = {(s < 0).mean():.1%}", fontsize=10)
    ax.set_yticks([]); ax.set_xlabel("коэффициент силуэта"); ax.set_xlim(-0.3, 0.8)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
for ax, k in zip(axes, [2, 3, 4, 5]):
    silhouette_plot(ax, Z, km_models[k].labels_, f"K-means, k={k}")
plt.tight_layout()
plt.show()
''')
md("k_selection_desc")

md("dendro")
code(r'''
LINK = linkage(Z, method="ward")          # иерархическая агломеративная кластеризация, метрика Уорда
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [2.2, 1]})
dendrogram(LINK, truncate_mode="lastp", p=30, show_leaf_counts=True, leaf_rotation=90,
           leaf_font_size=9, color_threshold=LINK[-2, 2], ax=axes[0])
cut = (LINK[-2, 2] + LINK[-3, 2]) / 2
axes[0].axhline(cut, ls="--", color="black")
axes[0].text(axes[0].get_xlim()[1] * 0.75, cut + 3, "разрез на 3 кластера")
axes[0].set_title("Дендрограмма (Ward, последние 30 объединений)")
axes[0].set_ylabel("расстояние объединения")
heights = LINK[::-1, 2][:15]
axes[1].plot(range(1, 16), heights, "o-", color="purple")
axes[1].set_xticks(range(1, 16))
axes[1].set_xlabel("число кластеров после разреза")
axes[1].set_ylabel("высота объединения")
axes[1].set_title("Высоты последних объединений")
plt.tight_layout()
plt.show()
gaps = pd.DataFrame({"k": range(1, 15), "высота слияния в k кластеров": heights[:14],
                     "скачок до k+1": -np.diff(heights)})
display(gaps.head(9).set_index("k"))

hc_rows = []
for k in K_RANGE:
    lab = fcluster(LINK, k, criterion="maxclust") - 1
    hc_rows.append({"k": k, "silhouette": silhouette_score(Z, lab),
                    "Calinski–Harabasz": calinski_harabasz_score(Z, lab),
                    "Davies–Bouldin": davies_bouldin_score(Z, lab),
                    "размеры кластеров": sorted(np.bincount(lab).tolist(), reverse=True)})
hc_sel = pd.DataFrame(hc_rows).set_index("k")
display(hc_sel)
''')
md("dendro_desc")

md("fit")
code(r'''
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)

# Перенумеруем кластеры по возрастанию доли просрочек – чтобы номера в обоих методах были сопоставимы
def relabel_by_risk(labels):
    order = pd.Series(y).groupby(labels).mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels)

lab_km = relabel_by_risk(km.labels_)
lab_hc = relabel_by_risk(hc.labels_)
print("Размеры кластеров K-means:", np.bincount(lab_km))
print("Размеры кластеров Ward   :", np.bincount(lab_hc))
print("Совпадение c fcluster(scipy):", adjusted_rand_score(lab_hc, fcluster(LINK, K, "maxclust")))
''')

code(r'''
def centers_of(A, labels):
    return np.vstack([A[labels == c].mean(axis=0) for c in np.unique(labels)])

def geometry(A, labels):
    C = centers_of(A, labels)
    intra = [np.linalg.norm(A[labels == c] - C[c], axis=1) for c in range(len(C))]
    D = cdist(C, C)
    np.fill_diagonal(D, np.nan)
    return C, intra, D

def pair_jaccard(a, b):
    (tn, fp), (fn, tp) = pair_confusion_matrix(a, b)
    return tp / (tp + fp + fn)

def all_metrics(A, labels, y_true):
    C, intra, D = geometry(A, labels)
    h, c, v = homogeneity_completeness_v_measure(y_true, labels)
    mean_intra = np.mean([d.mean() for d in intra])
    return {
        "число кластеров": len(C),
        "silhouette (↑)": silhouette_score(A, labels),
        "Calinski–Harabasz (↑)": calinski_harabasz_score(A, labels),
        "Davies–Bouldin (↓)": davies_bouldin_score(A, labels),
        "WCSS": sum(((A[labels == k] - C[k]) ** 2).sum() for k in range(len(C))),
        "ср. расстояние до центра": mean_intra,
        "мин. расстояние между центрами": np.nanmin(D),
        "ср. расстояние между центрами": np.nanmean(D),
        "отношение межкл./внутрикл.": np.nanmin(D) / mean_intra,
        "Rand index (↑)": rand_score(y_true, labels),
        "Adjusted Rand (↑)": adjusted_rand_score(y_true, labels),
        "Jaccard (пары, ↑)": pair_jaccard(y_true, labels),
        "Fowlkes–Mallows (↑)": fowlkes_mallows_score(y_true, labels),
        "NMI (↑)": normalized_mutual_info_score(y_true, labels),
        "однородность": h, "полнота": c, "V-мера": v,
    }

metrics = pd.DataFrame({"K-means (k=3)": all_metrics(Z, lab_km, y),
                        "Иерархическая Ward (k=3)": all_metrics(Z, lab_hc, y)})
# опорные значения: случайное разбиение с теми же размерами кластеров
rng = np.random.default_rng(SEED)
metrics["случайные метки (размеры как у K-means)"] = pd.Series(all_metrics(Z, rng.permutation(lab_km), y))
display(metrics.style.format("{:.3f}"))
''')
md("metrics_desc")

md("geometry")
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, (name, lab) in zip(axes[:2], [("K-means", lab_km), ("Ward", lab_hc)]):
    C, intra, D = geometry(Z, lab)
    Dshow = cdist(C, C)
    sns.heatmap(Dshow, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False,
                xticklabels=[f"C{i}" for i in range(K)], yticklabels=[f"C{i}" for i in range(K)])
    ax.set_title(f"{name}: расстояния между центрами")
geo = []
for name, lab in [("K-means", lab_km), ("Ward", lab_hc)]:
    C, intra, D = geometry(Z, lab)
    for c in range(K):
        geo.append({"метод": name, "кластер": f"C{c}", "размер": len(intra[c]),
                    "ср. расстояние до центра": intra[c].mean(),
                    "медиана": np.median(intra[c]), "радиус (95%)": np.quantile(intra[c], .95),
                    "ближайший другой центр": np.nanmin(D[c])})
geo = pd.DataFrame(geo)
sns.barplot(data=geo, x="кластер", y="ср. расстояние до центра", hue="метод", ax=axes[2])
axes[2].set_title("Компактность: среднее расстояние объектов до центра")
plt.tight_layout()
plt.show()
display(geo.set_index(["метод", "кластер"]))
''')
md("geometry_desc")

md("external")
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, (name, lab) in zip(axes[:2], [("K-means", lab_km), ("Ward", lab_hc)]):
    ct = pd.crosstab(pd.Series(lab, name="кластер"), pd.Series(y, name="dlq_2yrs"))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Oranges", ax=ax, cbar=False)
    ax.set_title(f"{name}: кластеры × класс")
risk = pd.DataFrame({"K-means": pd.Series(y).groupby(lab_km).mean(),
                     "Ward": pd.Series(y).groupby(lab_hc).mean()})
risk.index = [f"C{i}" for i in risk.index]
risk.plot.bar(ax=axes[2], rot=0, color=["steelblue", "darkorange"])
axes[2].axhline(y.mean(), ls="--", color="grey", label="в среднем")
axes[2].set_title("Доля заёмщиков с просрочкой в кластере"); axes[2].legend()
plt.tight_layout()
plt.show()
display(risk.rename_axis("кластер").style.format("{:.3f}"))
''')
md("external_desc")

md("params")
code(r'''
rows = []
for k in K_RANGE:
    for init in ["k-means++", "random"]:
        for n_init in [1, 10]:
            m = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=SEED).fit(Z)
            rows.append({"k": k, "init": init, "n_init": n_init, "WCSS": m.inertia_, "итераций": m.n_iter_,
                         "silhouette": silhouette_score(Z, m.labels_, sample_size=8000, random_state=SEED),
                         "ARI с классом": adjusted_rand_score(y, m.labels_)})
grid = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
for (init, n_init), g in grid.groupby(["init", "n_init"]):
    lbl = f"{init}, n_init={n_init}"
    axes[0].plot(g["k"], g["WCSS"], "o-", label=lbl)
    axes[1].plot(g["k"], g["silhouette"], "o-", label=lbl)
    axes[2].plot(g["k"], g["ARI с классом"], "o-", label=lbl)
for ax, t in zip(axes, ["WCSS", "silhouette", "ARI с классом dlq_2yrs"]):
    ax.set_title(f"K-means: {t} от k"); ax.set_xlabel("k"); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
display(grid.pivot_table(index="k", columns=["init", "n_init"], values="WCSS").round(0))
''')
code(r'''
# Устойчивость к начальной инициализации: 20 запусков с n_init=1 при разных seed
stab = []
for k in [2, 3, 4, 5, 6]:
    labs = [KMeans(n_clusters=k, n_init=1, init="random", random_state=s).fit(Z) for s in range(20)]
    inert = np.array([m.inertia_ for m in labs])
    ref = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z).labels_
    aris = [adjusted_rand_score(ref, m.labels_) for m in labs]
    stab.append({"k": k, "WCSS: мин": inert.min(), "WCSS: макс": inert.max(),
                 "разброс WCSS, %": (inert.max() - inert.min()) / inert.min() * 100,
                 "ARI с эталоном: среднее": np.mean(aris), "ARI: мин": np.min(aris)})
display(pd.DataFrame(stab).set_index("k"))
''')
md("params_kmeans_desc")

code(r'''
link_rows = []
for method in ["ward", "complete", "average", "single"]:
    L = LINK if method == "ward" else linkage(Z, method=method)
    for k in [2, 3, 4, 5]:
        lab = fcluster(L, k, criterion="maxclust") - 1
        link_rows.append({"связь": method, "k": k,
                          "silhouette": silhouette_score(Z, lab, sample_size=8000, random_state=SEED),
                          "ARI с классом": adjusted_rand_score(y, lab),
                          "размеры кластеров": sorted(np.bincount(lab).tolist(), reverse=True)})
display(pd.DataFrame(link_rows).set_index(["связь", "k"]))
''')
md("params_linkage_desc")

code(r'''
# DBSCAN: k-distance график для выбора eps и сетка eps × min_samples
fig, ax = plt.subplots(figsize=(8, 4))
for ms in [10, 20, 40]:
    d = np.sort(NearestNeighbors(n_neighbors=ms).fit(Z).kneighbors(Z)[0][:, -1])
    ax.plot(d, label=f"min_samples={ms}")
ax.set_ylim(0, 4); ax.set_xlabel("объекты, отсортированные по расстоянию"); ax.set_ylabel("расстояние до k-го соседа")
ax.set_title("k-distance график для DBSCAN"); ax.legend()
plt.tight_layout()
plt.show()

db_rows = []
for eps in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
    for ms in [10, 20, 40]:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
        n_cl = len(set(lab)) - (1 if -1 in lab else 0)
        mask = lab != -1
        sil = silhouette_score(Z[mask], lab[mask], sample_size=min(8000, mask.sum()), random_state=SEED) \
            if n_cl >= 2 else np.nan
        sizes = np.bincount(lab[mask]) if mask.any() else np.array([0])
        db_rows.append({"eps": eps, "min_samples": ms, "кластеров": n_cl, "доля шума": 1 - mask.mean(),
                        "доля крупнейшего кластера": sizes.max() / len(lab), "silhouette (без шума)": sil})
db = pd.DataFrame(db_rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.heatmap(db.pivot(index="min_samples", columns="eps", values="кластеров"), annot=True, fmt="d", cmap="Purples", ax=axes[0])
axes[0].set_title("DBSCAN: число кластеров")
sns.heatmap(db.pivot(index="min_samples", columns="eps", values="доля шума"), annot=True, fmt=".2f", cmap="Reds", ax=axes[1])
axes[1].set_title("DBSCAN: доля шума")
plt.tight_layout()
plt.show()
display(db.set_index(["eps", "min_samples"]))
''')
md("params_dbscan_desc")

# =================================================================== 5. Визуализация
md("viz")
code(r'''
pca2 = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca2.transform(Z)
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
pal = sns.color_palette("Set1", K)
for ax, (name, lab) in zip(axes[:2], [("K-means", lab_km), ("Иерархическая (Ward)", lab_hc)]):
    for c in range(K):
        m = lab == c
        ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.35, color=pal[c], label=f"C{c} (n={m.sum()})")
    Cp = pca2.transform(centers_of(Z, lab))
    ax.scatter(Cp[:, 0], Cp[:, 1], marker="X", s=250, c="black", edgecolor="white", linewidth=1.5, label="центры")
    ax.set_title(f"{name}, k={K}")
for cls, color in [(0, "seagreen"), (1, "crimson")]:
    m = y == cls
    axes[2].scatter(P[m, 0], P[m, 1], s=4, alpha=0.3, color=color, label=f"dlq_2yrs={cls}")
axes[2].set_title("Истинный класс (для сравнения)")
for ax in axes:
    leg = ax.legend(fontsize=9, loc="upper right")
    for h in leg.legend_handles:
        h.set_sizes([40]); h.set_alpha(1)
for ax in axes:
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
plt.tight_layout()
plt.show()
''')
code(r'''
# Кластеры в исходных признаках: несколько наиболее «разделяющих» пар
pairs = [("late_30_59", "rev_util"), ("age", "rev_util"), ("monthly_inc", "debt_ratio"), ("open_credit", "real_estate")]
sample_idx = np.random.default_rng(SEED).choice(len(Z), 4000, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for row, (name, lab) in enumerate([("K-means", lab_km), ("Ward", lab_hc)]):
    for ax, (a, b) in zip(axes[row], pairs):
        jit = np.random.default_rng(1).normal(0, 0.04, size=(len(sample_idx), 2))
        ax.scatter(X_t[a].values[sample_idx] + jit[:, 0], X_t[b].values[sample_idx] + jit[:, 1],
                   c=[pal[c] for c in lab[sample_idx]], s=5, alpha=0.4)
        ax.set_xlabel(a + (" (log1p)" if a in LOG_COLS else "")); ax.set_ylabel(b + (" (log1p)" if b in LOG_COLS else ""))
        ax.set_title(f"{name}: {a} × {b}", fontsize=10)
plt.tight_layout()
plt.show()
''')
md("viz_desc")

md("profiles")
code(r'''
def profile(lab):
    p = X_raw.groupby(lab).median()
    p.insert(0, "размер", np.bincount(lab))
    p["ср. число просрочек"] = X_raw[late_cols].sum(axis=1).groupby(lab).mean()
    p["доля без просрочек"] = (X_raw[late_cols].sum(axis=1) == 0).groupby(lab).mean()
    p["доля dlq_2yrs=1"] = pd.Series(y).groupby(lab).mean()
    p.index = [f"C{i}" for i in p.index]
    return p

prof_km, prof_hc = profile(lab_km), profile(lab_hc)
print("K-means: медианы признаков в исходных единицах")
display(prof_km.style.format("{:.2f}"))
print("Ward: медианы признаков в исходных единицах")
display(prof_hc.style.format("{:.2f}"))

fig, axes = plt.subplots(1, 2, figsize=(20, 4.2))
for ax, (name, lab) in zip(axes, [("K-means", lab_km), ("Ward", lab_hc)]):
    C = pd.DataFrame(centers_of(Z, lab), columns=FEATURES, index=[f"C{i}" for i in range(K)])
    sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, ax=ax)
    ax.set_title(f"{name}: центры кластеров (z-оценки)")
plt.tight_layout()
plt.show()
''')
md("profiles_desc")

md("compare")
code(r'''
ct = pd.crosstab(pd.Series(lab_km, name="K-means"), pd.Series(lab_hc, name="Ward"))
print(f"Согласованность методов: ARI = {adjusted_rand_score(lab_km, lab_hc):.3f}, "
      f"NMI = {normalized_mutual_info_score(lab_km, lab_hc):.3f}, "
      f"доля объектов в «одноимённых» кластерах = {(lab_km == lab_hc).mean():.3f}")
display(ct)

# Устойчивость на подвыборках: 15 раз берём 70% объектов, повторяем кластеризацию
# и сравниваем с исходными метками на тех же объектах
boot = []
rng = np.random.default_rng(SEED)
for b in range(15):
    idx = rng.choice(len(Z), int(0.7 * len(Z)), replace=False)
    l_km = KMeans(n_clusters=K, n_init=10, random_state=b).fit_predict(Z[idx])
    l_hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z[idx])
    boot.append({"K-means": adjusted_rand_score(lab_km[idx], l_km), "Ward": adjusted_rand_score(lab_hc[idx], l_hc)})
boot = pd.DataFrame(boot)
display(boot.describe().loc[["mean", "std", "min", "max"]].rename(index={"mean": "среднее ARI", "std": "СКО", "min": "мин", "max": "макс"}))

cmp_cols = ["silhouette (↑)", "Calinski–Harabasz (↑)", "Davies–Bouldin (↓)", "Adjusted Rand (↑)", "Jaccard (пары, ↑)", "NMI (↑)"]
fig, axes = plt.subplots(1, len(cmp_cols) + 1, figsize=(22, 3.8))
for ax, col in zip(axes, cmp_cols):
    metrics.loc[col, ["K-means (k=3)", "Иерархическая Ward (k=3)"]].plot.bar(ax=ax, color=["steelblue", "darkorange"], rot=0)
    ax.set_title(col, fontsize=10); ax.set_xticklabels(["K-means", "Ward"])
boot.plot.box(ax=axes[-1]); axes[-1].set_title("Устойчивость (ARI на подвыборках)", fontsize=10)
plt.tight_layout()
plt.show()
''')
md("compare_desc")

# =================================================================== 6-8
md("conclusion")
md("sources")
md("appendix")
code(r'''
import sys, sklearn, scipy, matplotlib
print("Python", sys.version.split()[0])
for lib in [np, pd, sklearn, scipy, matplotlib, sns]:
    print(f"{lib.__name__:12s} {lib.__version__}")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "lab3.ipynb")
print(f"lab3.ipynb: {len(cells)} ячеек")
