import numpy  as np
import pandas as pd
import os

from data_fetcher import fetch_games_data

# ──────────────────────────────────────────────
#  КОНСТАНТЫ
# ──────────────────────────────────────────────
IQR_MULTIPLIER     = 1.5   # стандартный порог выброса по IQR
FILL_TEXT_UNKNOWN  = "Неизвестно"
TOP_N              = 5
OUTPUT_DIR         = "output"
DATA_DIR           = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ══════════════════════════════════════════════
#  ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Загружает датасет через data_fetcher.

    Приоритет:
      1. data/videogames.csv (кэш)
      2. RAWG API + SteamSpy (если задан RAWG_API_KEY)
      3. Синтетический генератор (резерв)

    Returns:
        pd.DataFrame: датафрейм с данными о видеоиграх.
    """
    df = fetch_games_data(n=300)
    print(f"[Загрузка] {len(df)} строк, {len(df.columns)} столбцов")
    return df


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №5 — Первый взгляд на DataFrame
# ──────────────────────────────────────────────
def task5_first_look(df: pd.DataFrame) -> None:
    """Выводит базовую информацию о датафрейме."""
    print("\n  ЗАДАНИЕ №5 — Первый взгляд на DataFrame")

    print(f"\ndf.head(10):\n{df.head(10).to_string()}")
    print(f"\ndf.tail(5):\n{df.tail(5).to_string()}")
    print(f"\ndf.shape  : {df.shape}  ← (строк, столбцов)")
    print(f"\ndf.dtypes :\n{df.dtypes}")
    print(f"\ndf.info() :")
    df.info()
    print(f"\ndf.describe():\n{df.describe().round(2).to_string()}")

    print("\n▸ Вывод: датасет содержит информацию о видеоиграх за 2018–2025.")
    print(f"  {df.shape[0]} строк, {df.shape[1]} столбцов.")
    print("  Числовые поля: бюджет, выручка, оценки, продажи, цена, онлайн.")
    print("  Категориальные: жанр, разработчик, издатель, платформа, тир.")

    # Дополнительная информация об источнике данных
    if "steam_app_id" in df.columns:
        real_count = df["steam_app_id"].notna().sum()
        if real_count > 0:
            print(f"\n  ★ Реальных игр со Steam App ID: {real_count}")
            print(f"  ★ Metacritic-оценок: {df['critic_score'].notna().sum()}")


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №6 — Пропущенные значения
# ──────────────────────────────────────────────
def task6_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает пропущенные значения в датафрейме.

    Args:
        df: исходный датафрейм.
    Returns:
        pd.DataFrame: очищенный датафрейм.
    """
    print("\n  ЗАДАНИЕ №6 — Пропущенные значения")

    print(f"\nДо очистки: shape = {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nПропуски по столбцам:\n{missing.to_string()}")
    else:
        print("\nПропусков нет — данные уже очищены в data_fetcher.")

    df = df.copy()

    # Числовые — заполнить медианой
    numeric_fill = ["budget_mln", "user_score", "peak_online",
                    "revenue_mln", "sales_mln", "positive_pct"]
    for col in numeric_fill:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"\n  [{col}] заполнен медианой = {median_val:.2f}")
            print("   Обоснование: медиана устойчива к выбросам.")

    # Текстовые — привести к object, затем заполнить «Неизвестно»
    # (колонки могут быть float64 если все значения NaN — RAWG не даёт publisher/developer)
    for col in ["developer", "publisher"]:
        if col in df.columns:
            df[col] = df[col].astype(object)   # float64 → object
            if df[col].isnull().any():
                df[col] = df[col].fillna(FILL_TEXT_UNKNOWN)
                print(f"\n  [{col}] заполнен '{FILL_TEXT_UNKNOWN}'")

    print(f"\nПосле очистки: shape = {df.shape}")
    print(f"Оставшихся пропусков: {df.isnull().sum().sum()}")
    return df


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №7 — Дубликаты и выбросы
# ──────────────────────────────────────────────
def task7_duplicates_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет дубликаты и обрабатывает выбросы методом IQR.

    Args:
        df: датафрейм.
    Returns:
        pd.DataFrame: очищенный датафрейм.
    """
    print("\n  ЗАДАНИЕ №7 — Дубликаты и выбросы (IQR)")

    df = df.copy()

    dup_count = df.duplicated().sum()
    print(f"\nДубликатов найдено: {dup_count}")
    df.drop_duplicates(inplace=True)
    print(f"После удаления дубликатов: {df.shape[0]} строк")

    print(f"\n{'Столбец':<20} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lo':>8} {'Hi':>8} {'Выбр.':>7}")
    print("-" * 68)

    numeric_cols = ["budget_mln", "revenue_mln", "critic_score",
                    "user_score", "sales_mln", "peak_online"]
    total_removed = 0

    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - IQR_MULTIPLIER * iqr
        hi  = q3 + IQR_MULTIPLIER * iqr
        mask    = (df[col] < lo) | (df[col] > hi)
        n_out   = mask.sum()
        total_removed += n_out
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"{col:<20} {q1:>8.1f} {q3:>8.1f} {iqr:>8.1f} {lo:>8.1f} {hi:>8.1f} {n_out:>7}")

    print(f"\nВсего скорректировано выбросов: {total_removed} (заменены границами IQR)")
    print("Обоснование: полное удаление строк сократило бы выборку; лучше сгладить.")
    print(f"\ndf.shape после: {df.shape}")
    return df


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №8 — Новые признаки и типы
# ──────────────────────────────────────────────
def task8_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт новые столбцы, проверяет и исправляет типы данных.

    Args:
        df: датафрейм.
    Returns:
        pd.DataFrame: датафрейм с новыми признаками.
    """
    print("\n  ЗАДАНИЕ №8 — Новые признаки и преобразование типов")

    df = df.copy()

    # Новые числовые столбцы
    df["profit_mln"]    = df["revenue_mln"] - df["budget_mln"]
    df["roi_pct"]       = (df["profit_mln"] / df["budget_mln"].replace(0, np.nan)) * 100
    df["score_gap"]     = df["critic_score"] - df["user_score"]   # >0: критики выше
    df["score_gap_abs"] = df["score_gap"].abs()

    print("\nНовые столбцы:")
    print(f"  profit_mln  = revenue - budget → первые 5: {df['profit_mln'].head().round(2).tolist()}")
    print(f"  roi_pct     = profit / budget  → первые 5: {df['roi_pct'].head().round(1).tolist()}")
    print(f"  score_gap   = critic - user    → первые 5: {df['score_gap'].head().round(1).tolist()}")
    print(f"  score_gap_abs                  → первые 5: {df['score_gap_abs'].head().round(1).tolist()}")

    # Дополнительный признак: источник данных (реальный / синтетический)
    if "steam_app_id" in df.columns:
        df["is_real_data"] = df["steam_app_id"].notna()
        real_n = df["is_real_data"].sum()
        print(f"\n  is_real_data: {real_n} реальных игр, {len(df)-real_n} синтетических")

    # Рейтинговый тир по оценке критиков
    df["critic_tier"] = pd.cut(
        df["critic_score"],
        bins=[0, 49, 64, 74, 84, 100],
        labels=["Провал (<50)", "Слабо (50–64)", "Средне (65–74)",
                "Хорошо (75–84)", "Шедевр (85+)"],
    )
    print(f"\n  critic_tier (по Metacritic):\n{df['critic_tier'].value_counts().to_string()}")

    # Типы
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["price_usd"]    = pd.to_numeric(df["price_usd"],    errors="coerce")

    # Переименование
    df.rename(columns={
        "budget_mln":   "Бюджет_млн",
        "revenue_mln":  "Выручка_млн",
        "profit_mln":   "Прибыль_млн",
        "critic_score": "Оц_критики",
        "user_score":   "Оц_игроки",
        "sales_mln":    "Продажи_млн",
        "score_gap":    "Разрыв_оценок",
    }, inplace=True)

    print(f"\ndf.dtypes после преобразований:\n{df.dtypes.to_string()}")
    return df


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №9 — GroupBy и сводные таблицы
# ──────────────────────────────────────────────
def task9_groupby(df: pd.DataFrame) -> None:
    """
    Группирует данные по жанру и другим категориям, строит сводную таблицу.

    Args:
        df: датафрейм (после feature engineering).
    """
    print("\n  ЗАДАНИЕ №9 — GroupBy и сводные таблицы")
    oc = "Оц_критики" if "Оц_критики" in df.columns else "critic_score"
    ou = "Оц_игроки"  if "Оц_игроки"  in df.columns else "user_score"
    pr = "Прибыль_млн" if "Прибыль_млн" in df.columns else "profit_mln"
    sa = "Продажи_млн" if "Продажи_млн" in df.columns else "sales_mln"

    # GroupBy по жанру
    group_genre = df.groupby("genre").agg(
        Кол_игр    = (oc, "count"),
        Ср_Оц_кр  = (oc, "mean"),
        Ср_Оц_игр = (ou, "mean"),
        Ср_Продажи= (sa, "mean"),
        Сум_Приб   = (pr, "sum"),
    ).round(2)

    print(f"\nСтатистика по жанрам:\n{group_genre.sort_values('Сум_Приб', ascending=False).to_string()}")

    # Несколько агрегаций
    agg_multi = df.groupby("tier").agg({oc: ["mean", "min", "max"], sa: ["mean", "sum"]}).round(2)
    print(f"\nАгрегация по тиру (AAA/AA/Indie):\n{agg_multi.to_string()}")

    # Сводная таблица: жанр × тир → средние продажи
    pivot = pd.pivot_table(
        df, values=sa, index="genre", columns="tier",
        aggfunc="mean", fill_value=0
    ).round(2)
    print(f"\nСводная таблица (средние продажи, млн), жанр × тир:\n{pivot.to_string()}")

    # Топ-5 по прибыли
    top5 = group_genre["Сум_Приб"].nlargest(TOP_N)
    print(f"\nТоп-{TOP_N} жанров по суммарной прибыли:\n{top5.to_string()}")
    print(f"\n▸ Вывод: жанр «{top5.index[0]}» приносит наибольшую совокупную прибыль.")

    # Дополнительно: анализ разрыва оценок по годам (если есть реальные данные)
    if "release_year" in df.columns and "Разрыв_оценок" in df.columns:
        gap_by_year = df.groupby("release_year")["Разрыв_оценок"].mean().round(2)
        print(f"\nСредний разрыв оценок (критики − игроки) по годам:\n{gap_by_year.to_string()}")


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №10 — Сортировка, фильтрация, query
# ──────────────────────────────────────────────
def task10_filter_sort(df: pd.DataFrame) -> None:
    """
    Фильтрует, сортирует, применяет query и случайную выборку.

    Args:
        df: датафрейм.
    """
    print("\n  ЗАДАНИЕ №10 — Сортировка, фильтрация и query")

    oc = "Оц_критики" if "Оц_критики" in df.columns else "critic_score"
    ou = "Оц_игроки"  if "Оц_игроки"  in df.columns else "user_score"
    pr = "Прибыль_млн" if "Прибыль_млн" in df.columns else "profit_mln"

    # Мульти-фильтр: критики ≥ 70, игроки < 50, AAA
    filt = df[(df[oc] >= 70) & (df[ou] < 50) & (df["tier"] == "AAA")]
    print(f"\nФильтр (критики≥70 И игроки<50 И AAA): {len(filt)} игр")
    print(f"  Это «провальные» ААА по мнению игроков с высокими оценками критиков.")
    if not filt.empty:
        print(filt[["title", oc, ou, "genre", pr]].head().to_string(index=False))

    # Сортировка по двум столбцам
    sorted_df = df.sort_values(by=[oc, ou], ascending=[False, True])
    print(f"\nСортировка (критики↓, игроки↑) — топ-5:")
    print(sorted_df[["title", oc, ou]].head().to_string(index=False))

    # .query()
    q_result = df.query(f"`{oc}` > 80 and `{ou}` > 80")
    print(f"\n.query('критики>80 AND игроки>80'): {len(q_result)} игр")
    print(f"  (Те же результаты через булеву маску: "
          f"{len(df[(df[oc]>80) & (df[ou]>80)])} игр)")

    # Случайная выборка
    sample_n = min(50, len(df))
    sample = df.sample(n=sample_n, random_state=1)
    print(f"\nСлучайная выборка (n={sample_n}):")
    print(f"  Среднее {oc}: выборка={sample[oc].mean():.2f}, полный={df[oc].mean():.2f}")
    print(f"  Среднее {ou}: выборка={sample[ou].mean():.2f}, полный={df[ou].mean():.2f}")
    print("  Вывод: выборка хорошо отражает полный датасет.")

    # Экстремумы
    worst_user = df.loc[df[ou].idxmin()]
    best_user  = df.loc[df[ou].idxmax()]
    print(f"\nИгра с ХУДШЕЙ  оценкой игроков : {worst_user['title']}  ({worst_user[ou]:.1f})")
    print(f"Игра с ЛУЧШЕЙ  оценкой игроков : {best_user['title']}   ({best_user[ou]:.1f})")


# ══════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════
def main() -> None:
    """Запускает все задания Части 2."""
    print("  ЧАСТЬ 2 — Pandas: загрузка, очистка, трансформация")
    print("  Тема: Факторы успеха и провала видеоигр (2018–2025)")

    df = load_data()

    task5_first_look(df)

    df = task6_missing_values(df)
    df = task7_duplicates_and_outliers(df)
    df = task8_feature_engineering(df)

    task9_groupby(df)
    task10_filter_sort(df)

    print("\n  ✓ Часть 2 выполнена успешно.")



if __name__ == "__main__":
    main()