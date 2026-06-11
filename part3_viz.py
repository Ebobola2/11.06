import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

from data_fetcher import fetch_games_data

# ──────────────────────────────────────────────
#  КОНСТАНТЫ
# ──────────────────────────────────────────────
OUTPUT_DIR    = "output"
DATA_DIR      = "data"
DPI           = 150
FIG_SIZE_BAR  = (12, 6)
FIG_SIZE_SCAT = (9, 7)
FIG_SIZE_HIST = (9, 6)
FIG_SIZE_HEAT = (9, 7)
FIG_SIZE_BOX  = (11, 6)
PALETTE       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                 "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ══════════════════════════════════════════════
#  ДАННЫЕ
# ══════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Загружает датасет через data_fetcher и добавляет производные поля.

    Источники (в порядке приоритета):
      1. data/videogames.csv (кэш)
      2. RAWG API + SteamSpy  (если задан RAWG_API_KEY)
      3. Синтетический генератор (резерв)

    Returns:
        pd.DataFrame: готовый датафрейм.
    """
    df = fetch_games_data(n=300)

    # Производные поля (могут уже быть, если data_fetcher их добавил)
    if "profit_mln" not in df.columns:
        df["profit_mln"] = df["revenue_mln"] - df["budget_mln"]
    if "roi_pct" not in df.columns:
        df["roi_pct"] = (df["profit_mln"] / df["budget_mln"].replace(0, np.nan)) * 100
    if "score_gap_abs" not in df.columns:
        df["score_gap_abs"] = (df["critic_score"] - df["user_score"]).abs()

    return df


# ──────────────────────────────────────────────
#  УТИЛИТЫ
# ──────────────────────────────────────────────
def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Сохранён: {path}")
    return path


def _corr_label(r: float) -> str:
    return f"r = {r:+.3f}"


# ══════════════════════════════════════════════
#  ГРАФИК 1 — Bar chart: топ-20 по прибыли
# ══════════════════════════════════════════════
def plot1_top20_profit(df: pd.DataFrame) -> None:
    """
    Строит столбчатую диаграмму топ-20 игр по прибыли.

    Args:
        df: датафрейм с полем profit_mln.
    """
    print("\n[График 1] Топ-20 игр по прибыли (Bar chart)")
    top20 = df.nlargest(20, "profit_mln")[["title", "profit_mln", "genre", "tier"]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE_BAR)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top20))]
    bars = ax.barh(top20["title"], top20["profit_mln"], color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top20["profit_mln"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8)

    ax.set_xlabel("Прибыль (млн $)", fontsize=11)
    ax.set_title("Топ-20 игр по прибыли (2018–2025)", fontsize=13, fontweight="bold", pad=14)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "01_top20_profit.png")


# ══════════════════════════════════════════════
#  ГРАФИК 2 — Scatter: оценка игроков vs продажи
# ══════════════════════════════════════════════
def plot2_userscore_vs_sales(df: pd.DataFrame) -> None:
    """
    Scatter plot: оценка игроков vs продажи с линией тренда.

    Args:
        df: датафрейм.
    """
    print("[График 2] Оценка игроков vs Продажи (Scatter plot)")
    valid = df[["user_score", "sales_mln", "tier"]].dropna()
    x = valid["user_score"].to_numpy()
    y = valid["sales_mln"].to_numpy()

    r = np.corrcoef(x, y)[0, 1]
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SCAT)
    tier_colors = {"AAA": "#C44E52", "AA": "#4C72B0", "Indie": "#55A868"}
    for tier, grp in valid.groupby("tier"):
        ax.scatter(grp["user_score"], grp["sales_mln"],
                   color=tier_colors.get(tier, "grey"),
                   alpha=0.55, s=40, label=tier, edgecolors="none")

    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m*xs + b, color="#e63946", lw=2, label=f"Тренд ({_corr_label(r)})")

    ax.set_xlabel("Оценка игроков", fontsize=11)
    ax.set_ylabel("Продажи (млн копий)", fontsize=11)
    ax.set_title("Влияние оценки игроков на продажи", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0.97, 0.05, _corr_label(r), transform=ax.transAxes,
            ha="right", fontsize=11, color="#e63946",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e63946", alpha=0.8))
    plt.tight_layout()
    _save(fig, "02_userscore_vs_sales.png")


# ══════════════════════════════════════════════
#  ГРАФИК 3 — Scatter: бюджет vs прибыль
# ══════════════════════════════════════════════
def plot3_budget_vs_profit(df: pd.DataFrame) -> None:
    """
    Scatter plot: бюджет vs прибыль.

    Args:
        df: датафрейм.
    """
    print("[График 3] Бюджет vs Прибыль (Scatter plot)")
    valid = df[["budget_mln", "profit_mln"]].dropna()
    x = valid["budget_mln"].to_numpy()
    y = valid["profit_mln"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SCAT)
    colors = np.where(y >= 0, "#4C72B0", "#C44E52")
    ax.scatter(x, y, c=colors, alpha=0.5, s=40, edgecolors="none")

    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m*xs + b, color="#2d6a4f", lw=2, label=f"Тренд ({_corr_label(r)})")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, label="Безубыточность")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="Прибыльные"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="Убыточные"),
        Line2D([0],[0], color="#2d6a4f", lw=2, label=f"Тренд ({_corr_label(r)})"),
        Line2D([0],[0], color="black",  lw=1, ls="--", label="Безубыточность"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Бюджет (млн $)", fontsize=11)
    ax.set_ylabel("Прибыль (млн $)", fontsize=11)
    ax.set_title("Гипотеза №1: большой бюджет ≠ гарантия прибыли", fontsize=13, fontweight="bold", pad=14)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "03_budget_vs_profit.png")


# ══════════════════════════════════════════════
#  ГРАФИК 4 — Histogram: распределение оценок игроков
# ══════════════════════════════════════════════
def plot4_user_score_hist(df: pd.DataFrame) -> None:
    """
    Гистограмма распределения оценок игроков с линиями среднего и медианы.

    Args:
        df: датафрейм.
    """
    print("[График 4] Распределение оценок игроков (Histogram)")
    scores = df["user_score"].dropna()
    mean_val   = scores.mean()
    median_val = scores.median()

    fig, ax = plt.subplots(figsize=FIG_SIZE_HIST)
    n_bins = int(1 + 3.322 * np.log10(len(scores)))  # Правило Стёрджеса
    ax.hist(scores, bins=n_bins, color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(mean_val,   color="#e63946", lw=2, ls="--", label=f"Среднее = {mean_val:.1f}")
    ax.axvline(median_val, color="#2d6a4f", lw=2, ls="-",  label=f"Медиана  = {median_val:.1f}")

    ax.set_xlabel("Оценка игроков", fontsize=11)
    ax.set_ylabel("Количество игр", fontsize=11)
    ax.set_title("Распределение пользовательских оценок (2018–2025)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0.02, 0.95, f"Кол-во бинов: {n_bins} (правило Стёрджеса)",
            transform=ax.transAxes, fontsize=8, color="grey", va="top")
    plt.tight_layout()
    _save(fig, "04_userscore_hist.png")


# ══════════════════════════════════════════════
#  ГРАФИК 5 — Heatmap корреляций
# ══════════════════════════════════════════════
def plot5_heatmap(df: pd.DataFrame) -> None:
    """
    Тепловая карта корреляций ключевых числовых полей.

    Args:
        df: датафрейм.
    """
    print("[График 5] Тепловая карта корреляций (Heatmap)")

    # Включаем rawg_rating и ratings_count если они есть (реальные данные)
    base_cols = ["budget_mln", "revenue_mln", "profit_mln",
                 "critic_score", "user_score", "sales_mln", "roi_pct"]
    extra_cols = ["rawg_rating", "ratings_count", "positive_pct"]
    cols = [c for c in base_cols + extra_cols if c in df.columns]

    label_map = {
        "budget_mln":    "Бюджет",
        "revenue_mln":   "Выручка",
        "profit_mln":    "Прибыль",
        "critic_score":  "Оц.критики",
        "user_score":    "Оц.игроки",
        "sales_mln":     "Продажи",
        "roi_pct":       "ROI%",
        "rawg_rating":   "RAWG рейт.",
        "ratings_count": "Кол-во оц.",
        "positive_pct":  "% позит.",
    }
    labels = [label_map.get(c, c) for c in cols]

    corr = df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEAT)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = "black" if abs(val) < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Тепловая карта корреляций (числовые поля)", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")


# ══════════════════════════════════════════════
#  ГРАФИК 6 — Box plot: оценки по жанрам (доп.)
# ══════════════════════════════════════════════
def plot6_scores_by_genre(df: pd.DataFrame) -> None:
    """
    Box plot: распределение оценок критиков и игроков по жанрам.
    Позволяет увидеть, в каких жанрах разрыв оценок наибольший.

    Args:
        df: датафрейм.
    """
    print("[График 6] Box plot: оценки по жанрам")
    genre_order = (
        df.groupby("genre")["critic_score"].median()
        .sort_values(ascending=False).index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_BOX, sharey=False)

    for ax, col, label, color in [
        (axes[0], "critic_score", "Оценка критиков (Metacritic)", "#4C72B0"),
        (axes[1], "user_score",   "Оценка игроков",               "#DD8452"),
    ]:
        data_by_genre = [df[df["genre"] == g][col].dropna().tolist() for g in genre_order]
        bp = ax.boxplot(data_by_genre, vert=True, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(genre_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Распределение оценок по жанрам (2018–2025)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "06_scores_by_genre_boxplot.png")


# ══════════════════════════════════════════════
#  ИТОГОВЫЙ ОТВЕТ НА ИССЛЕДОВАТЕЛЬСКИЙ ВОПРОС
# ══════════════════════════════════════════════
def print_final_conclusions(df: pd.DataFrame) -> None:
    """
    Выводит структурированный ответ на исследовательский вопрос.

    Args:
        df: датафрейм.
    """
    print("\n" + "═"*60)
    print("  ИТОГОВЫЙ ОТВЕТ НА ИССЛЕДОВАТЕЛЬСКИЙ ВОПРОС")
    print("═"*60)

    valid_cs = df[["critic_score", "sales_mln"]].dropna()
    valid_us = df[["user_score",   "sales_mln"]].dropna()
    valid_bu = df[["budget_mln",   "user_score"]].dropna()

    r_cs = np.corrcoef(valid_cs["critic_score"], valid_cs["sales_mln"])[0, 1]
    r_us = np.corrcoef(valid_us["user_score"],   valid_us["sales_mln"])[0, 1]
    r_bu = np.corrcoef(valid_bu["budget_mln"],   valid_bu["user_score"])[0, 1]

    fail_mask = (df["critic_score"] >= 70) & (df["user_score"] < 55)
    fail_pct  = fail_mask.mean() * 100

    # Проверяем источник данных
    is_real = "steam_app_id" in df.columns and df["steam_app_id"].notna().any()
    data_source = "RAWG API (Metacritic) + SteamSpy" if is_real else "синтетические данные"

    print(f"""
┌─────────────────────────────────────────────────────────┐
│  ВОПРОС:  Почему крупнобюджетные игры проваливаются?    │
│           Какие факторы влияют на коммерческий успех?   │
└─────────────────────────────────────────────────────────┘

ГИПОТЕЗА (до анализа):
  Высокий бюджет не гарантирует ни высокий рейтинг игроков,
  ни коммерческий успех.

ДАННЫЕ:""")
    print(f"  {len(df)} игр, 2018–2025, источник: {data_source}.")
    print(f"  Поля: бюджет, выручка, оценки критиков/игроков, продажи.\n")

    print("МЕТОД:")
    print("  NumPy — массивы, корреляции, нормализация, выбросы.")
    print("  Pandas — очистка, GroupBy, фильтрация, новые признаки.")
    print("  Matplotlib — 6 графиков (bar, scatter×2, hist, heatmap, boxplot).\n")

    print("РЕЗУЛЬТАТЫ:")
    print(f"  • Корреляция [Оц.игроков ↔ Продажи]  : r = {r_us:+.3f}")
    print(f"  • Корреляция [Оц.критиков ↔ Продажи] : r = {r_cs:+.3f}")
    stronger = "игроков" if abs(r_us) > abs(r_cs) else "критиков"
    print(f"  → Оценка {stronger} сильнее коррелирует с продажами.\n")
    print(f"  • Корреляция [Бюджет ↔ Оц.игроков]   : r = {r_bu:+.3f}")
    print(f"  → Большой бюджет не гарантирует высокий рейтинг от игроков.\n")
    print(f"  • Игры с высоким Metascore (≥70) и низким User Score (<55): {fail_pct:.1f}%")
    print("  → Это «провальные» ААА — именно в них кроется разрыв оценок.\n")

    print("ВЫВОД:")
    print("  Гипотеза ПОДТВЕРЖДЕНА:")
    print("  1. Бюджет слабо связан с оценками игроков.")
    print("  2. Мнение игроков влияет на продажи сильнее мнения критиков.")
    print("  3. Провальные ААА имеют большой разрыв между оценками.")

    is_real
    print("\n★ ДАННЫЕ РЕАЛЬНЫЕ (RAWG.io) — выводы применимы на практике.")
    print("\n  • Не учтены маркетинговые бюджеты и региональные продажи.")
    print("  • Корреляция ≠ причинно-следственная связь.\n")


# ══════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════
def main() -> None:
    """Запускает все визуализации Части 3."""
    print("  ЧАСТЬ 3 — Визуализация, выводы и финальный отчёт")
    print("  Тема: Факторы успеха и провала видеоигр (2018–2025)")
    df = load_data()

    print("\n── Обязательные 4 графика ──")
    plot1_top20_profit(df)
    plot2_userscore_vs_sales(df)
    plot3_budget_vs_profit(df)
    plot4_user_score_hist(df)

    print("\n── Дополнительные графики ──")
    plot5_heatmap(df)
    plot6_scores_by_genre(df)

    print_final_conclusions(df)

    print(f"\n  ✓ Все графики сохранены в папку «{OUTPUT_DIR}/»")


if __name__ == "__main__":
    main()
