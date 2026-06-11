import numpy as np
import os

# Импортируем общий модуль парсинга
from data_fetcher import fetch_games_data

# ──────────────────────────────────────────────
#  КОНСТАНТЫ
# ──────────────────────────────────────────────
OUTLIER_STD_MULTIPLIER = 2       # порог выброса: mean ± N*std
ROLLING_WINDOW         = 7       # окно скользящего среднего
PERCENTILES            = [25, 50, 75]
OUTPUT_DIR             = "output"
DATA_DIR               = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ══════════════════════════════════════════════
#  ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════
def load_numpy_data(n: int = 200) -> dict:
    """
    Загружает датасет (реальный или синтетический) и возвращает
    словарь NumPy-массивов по каждому числовому полю.

    Источники (в порядке приоритета):
      1. data/videogames.csv — если уже скачан
      2. RAWG API + SteamSpy  — если задан RAWG_API_KEY
      3. Синтетический генератор (резерв)

    Args:
        n: желаемое количество строк
    Returns:
        dict: словарь NumPy-массивов
    """
    df = fetch_games_data(n=n)

    # Обязательные поля — убираем строки с NaN в ключевых колонках
    required = ["critic_score", "user_score", "budget_mln", "sales_mln"]
    df_clean = df.dropna(subset=required)

    print(f"[Данные] {len(df_clean)} игр после удаления строк с пропусками в ключевых полях")
    print(f"  Источник: {'RAWG API + SteamSpy' if df_clean.get('steam_app_id', [None])[0] else 'синтетика/CSV'}")

    return {
        "critic_scores": df_clean["critic_score"].to_numpy(dtype=float),
        "user_scores":   df_clean["user_score"].to_numpy(dtype=float),
        "budgets_mln":   df_clean["budget_mln"].to_numpy(dtype=float),
        "sales_mln":     df_clean["sales_mln"].to_numpy(dtype=float),
    }


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №1 — Создание и изучение массивов
# ──────────────────────────────────────────────
def task1_array_basics(data: dict) -> None:
    """
    Загружает массивы, выводит shape / dtype / size, делает срезы и reshape.

    Args:
        data: словарь NumPy-массивов.
    """
    print("\n  ЗАДАНИЕ №1 — Создание и изучение массивов")

    arr = data["critic_scores"]

    print(f"\n[Массив: оценки критиков]")
    print(f"  shape : {arr.shape}   ← (N,) — одномерный массив из N элементов")
    print(f"  dtype : {arr.dtype}   ← тип данных каждого элемента")
    print(f"  size  : {arr.size}    ← общее число элементов")

    print(f"\nПервые 10 элементов    : {np.round(arr[:10], 1)}")
    print(f"Каждый 3-й (первые 9) : {np.round(arr[::3][:9], 1)}")
    print(f"С 50-го по 60-й       : {np.round(arr[50:60], 1)}")

    # reshape: (N,) → (20, -1)
    n_rows = 20
    n_cols = len(arr) // n_rows
    arr_slice = arr[:n_rows * n_cols]
    reshaped = arr_slice.reshape(n_rows, n_cols)
    print(f"\nreshape({n_rows}, {n_cols}) → shape: {reshaped.shape}")
    print("  Зачем: например, для вычисления средних по группам (строкам).")
    print(f"  Средние по группам (первые 5): {np.round(reshaped.mean(axis=1)[:5], 2)}")


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №2 — Описательная статистика
# ──────────────────────────────────────────────
def task2_descriptive_stats(data: dict) -> None:
    """
    Вычисляет описательную статистику для каждого числового столбца.

    Args:
        data: словарь NumPy-массивов.
    """
    print("\n  ЗАДАНИЕ №2 — Описательная статистика")

    header = f"{'Показатель':<22} {'Оц.критиков':>13} {'Оц.игроков':>12} {'Бюджет,млн$':>13} {'Продажи,млн':>13}"
    print("\n" + header)
    print("-" * len(header))

    labels = ["critic_scores", "user_scores", "budgets_mln", "sales_mln"]

    metrics: dict = {}
    for key in labels:
        arr = data[key]
        metrics[key] = {
            "min":    arr.min(),
            "max":    arr.max(),
            "mean":   arr.mean(),
            "median": np.median(arr),
            "std":    arr.std(),
            "var":    arr.var(),
            "p25":    np.percentile(arr, 25),
            "p75":    np.percentile(arr, 75),
            "argmin": int(np.argmin(arr)),
            "argmax": int(np.argmax(arr)),
        }

    rows = [
        ("Минимум",            "min"),
        ("Максимум",           "max"),
        ("Среднее",            "mean"),
        ("Медиана",            "median"),
        ("Ст. отклонение",     "std"),
        ("Дисперсия",          "var"),
        ("Перцентиль 25%",     "p25"),
        ("Перцентиль 75%",     "p75"),
        ("Индекс минимума",    "argmin"),
        ("Индекс максимума",   "argmax"),
    ]

    for name, key in rows:
        vals = [f"{metrics[col][key]:>13.2f}" for col in labels]
        print(f"{name:<22} {''.join(vals)}")

    print("\n▸ Вывод:")
    cs, us = metrics["critic_scores"], metrics["user_scores"]
    print(f"  Критики оценивают игры в среднем {cs['mean']:.1f}, игроки — {us['mean']:.1f}.")
    diff = cs["mean"] - us["mean"]
    sign = "выше" if diff > 0 else "ниже"
    print(f"  Критики ставят оценки {abs(diff):.1f} пунктов {sign} игроков.")
    print(f"  Std у игроков ({us['std']:.1f}) {'>' if us['std'] > cs['std'] else '<'} "
          f"std у критиков ({cs['std']:.1f}) →")
    print("  мнения игроков расходятся " + ("сильнее." if us['std'] > cs['std'] else "меньше."))
    print(f"\n  Std vs Дисперсия: дисперсия = std² ({cs['std']:.2f}² ≈ {cs['var']:.1f}).")
    print("  Std удобнее — в тех же единицах, что данные.")


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №3 — Фильтрация и булева индексация
# ──────────────────────────────────────────────
def task3_filtering(data: dict) -> None:
    """
    Фильтрует выбросы, создаёт метки «высокий»/«низкий», считает NaN.

    Args:
        data: словарь NumPy-массивов.
    """
    print("\n  ЗАДАНИЕ №3 — Фильтрация и булева индексация")

    arr  = data["user_scores"]
    mean = arr.mean()
    std  = arr.std()

    above_mean = arr[arr > mean]
    print(f"\nЗначения выше среднего ({mean:.1f}): {len(above_mean)} из {len(arr)}")
    print(f"  Первые 10: {np.round(above_mean[:10], 1)}")

    lo = mean - OUTLIER_STD_MULTIPLIER * std
    hi = mean + OUTLIER_STD_MULTIPLIER * std
    outliers = arr[(arr < lo) | (arr > hi)]
    print(f"\nВыбросы (mean ± {OUTLIER_STD_MULTIPLIER}·std): [{lo:.1f}, {hi:.1f}]")
    print(f"  Найдено выбросов: {len(outliers)}")
    if len(outliers):
        print(f"  Значения: {np.round(outliers[:10], 1)}")

    labels = np.where(arr > mean, "высокий", "низкий")
    high_count = (labels == "высокий").sum()
    print(f"\nnp.where — метка «высокий»: {high_count}, «низкий»: {len(arr)-high_count}")

    # Симулируем NaN для демонстрации (реальные данные уже очищены)
    arr_with_nan = arr.copy()
    nan_idx = np.random.default_rng(7).choice(len(arr), size=12, replace=False)
    arr_with_nan[nan_idx] = np.nan
    nan_ratio = np.isnan(arr_with_nan).mean() * 100
    print(f"\nДоля NaN (искусственно добавлено): {nan_ratio:.1f}%")


# ──────────────────────────────────────────────
#  ЗАДАНИЕ №4 — Математика и корреляция
# ──────────────────────────────────────────────
def task4_math_and_correlation(data: dict) -> None:
    """
    Нормализует массивы (min-max и Z), считает скользящее среднее и корреляцию.

    Args:
        data: словарь NumPy-массивов.
    """
    print("\n  ЗАДАНИЕ №4 — Математические операции и корреляция")
    arr = data["user_scores"]

    # Min-Max нормализация
    mn, mx = arr.min(), arr.max()
    minmax = (arr - mn) / (mx - mn)
    print(f"\nMin-Max нормализация → диапазон [{minmax.min():.3f}, {minmax.max():.3f}]")
    print(f"  Первые 5: {np.round(minmax[:5], 3)}")

    # Z-нормализация
    z_norm = (arr - arr.mean()) / arr.std()
    print(f"\nZ-нормализация → mean≈{z_norm.mean():.4f}, std≈{z_norm.std():.4f}")
    print(f"  Первые 5: {np.round(z_norm[:5], 3)}")

    print("\n▸ Когда что использовать:")
    print("  Min-Max → когда важен физический диапазон (напр., нейросети, KNN).")
    print("  Z-norm  → когда важно сравнивать отклонения (PCA, линейные модели).")

    # Скользящее среднее
    kernel  = np.ones(ROLLING_WINDOW) / ROLLING_WINDOW
    rolling = np.convolve(arr, kernel, mode="valid")
    print(f"\nСкользящее среднее (окно={ROLLING_WINDOW}): первые 5 = {np.round(rolling[:5], 2)}")
    print(f"  Длина после свёртки: {len(rolling)} (сократилась на {len(arr)-len(rolling)})")

    # Корреляция
    cs = data["critic_scores"]
    us = data["user_scores"]
    bu = data["budgets_mln"]
    sa = data["sales_mln"]

    r_cs_us = np.corrcoef(cs, us)[0, 1]
    r_cs_sa = np.corrcoef(cs, sa)[0, 1]
    r_us_sa = np.corrcoef(us, sa)[0, 1]
    r_bu_sa = np.corrcoef(bu, sa)[0, 1]
    r_bu_us = np.corrcoef(bu, us)[0, 1]

    print("\n▸ Корреляции (Пирсон):")
    print(f"  Оц.критиков ↔ Оц.игроков : r = {r_cs_us:+.3f}")
    print(f"  Оц.критиков ↔ Продажи    : r = {r_cs_sa:+.3f}")
    print(f"  Оц.игроков  ↔ Продажи    : r = {r_us_sa:+.3f}")
    print(f"  Бюджет      ↔ Продажи    : r = {r_bu_sa:+.3f}")
    print(f"  Бюджет      ↔ Оц.игроков : r = {r_bu_us:+.3f}")

    abs_cs = abs(r_cs_sa)
    abs_us = abs(r_us_sa)
    stronger = "игроков" if abs_us > abs_cs else "критиков"
    print(f"\n▸ Вывод: оценка {stronger} сильнее коррелирует с продажами.")
    print(f"  (|r_user|={abs_us:.3f}  vs  |r_critic|={abs_cs:.3f})")


# ══════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════
def main() -> None:
    """Запускает все задания Части 1."""
    print("  ЧАСТЬ 1 — NumPy: массивы, математика, статистика")
    print("  Тема: Факторы успеха и провала видеоигр (2018–2025)")

    data = load_numpy_data(n=200)

    task1_array_basics(data)
    task2_descriptive_stats(data)
    task3_filtering(data)
    task4_math_and_correlation(data)

    print("\n  ✓ Часть 1 выполнена успешно.")


if __name__ == "__main__":
    main()
