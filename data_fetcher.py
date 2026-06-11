"""
╔══════════════════════════════════════════════════════════════════╗
║  DATA FETCHER — Автоматический парсинг данных о видеоиграх       ║
║  Источники: RAWG API (metacritic + оценки) + Steam Store API     ║
║                                                                  ║
║  Использование:                                                  ║
║    from data_fetcher import fetch_games_data                     ║
║    df = fetch_games_data(n=200)                                  ║
║                                                                  ║
║  RAWG API:                                                       ║
║    • Бесплатный, официальный                                     ║
║    • Регистрация на rawg.io → получите API-ключ (бесплатно)      ║
║    • Содержит: Metacritic-рейтинг, жанры, платформы, дату релиза ║
║  Steam Store API:                                                ║
║    • Официальный, без ключа                                      ║
║    • Содержит: пиковый онлайн, цену, Steam-рейтинг               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import logging
import requests
import numpy  as np
import pandas as pd
from pathlib import Path
from typing  import Optional

# ──────────────────────────────────────────────
#  НАСТРОЙКИ
# ──────────────────────────────────────────────
DATA_DIR     = "data"
CSV_PATH     = os.path.join(DATA_DIR, "videogames.csv")
CACHE_PATH   = os.path.join(DATA_DIR, "rawg_cache.json")

# Ваш RAWG API-ключ. Можно задать через переменную окружения RAWG_API_KEY
# или вставить прямо сюда. Получите БЕСПЛАТНО на https://rawg.io/apidocs
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "80b6a5248e9f4b2a9a1a75a31a60d966")

RAWG_BASE    = "https://api.rawg.io/api"
STEAM_BASE   = "https://store.steampowered.com/api"
STEAM_SPY    = "https://steamspy.com/api.php"

YEARS_RANGE  = (2018, 2025)
PAGE_SIZE    = 40         # RAWG: max 40
SLEEP_SEC    = 0.35       # пауза между запросами (соблюдаем rate limit)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════
#  RAWG API
# ══════════════════════════════════════════════

def _rawg_get(endpoint: str, params: dict) -> Optional[dict]:
    """
    Выполняет GET-запрос к RAWG API.

    Args:
        endpoint: путь (напр. '/games')
        params:   query-параметры
    Returns:
        JSON-ответ или None при ошибке
    """
    if RAWG_API_KEY == "YOUR_API_KEY_HERE":
        return None   # ключ не настроен — пропускаем

    params["key"] = RAWG_API_KEY
    url = RAWG_BASE + endpoint
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"RAWG ошибка: {e}")
        return None


def fetch_rawg_games(n: int = 200) -> list[dict]:
    """
    Получает список игр из RAWG API с фильтром по годам и Metacritic-рейтингу.

    Args:
        n: желаемое количество игр
    Returns:
        Список словарей с полями игр
    """
    # Проверяем кэш
    if Path(CACHE_PATH).exists():
        log.info(f"Загружаем кэш RAWG: {CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= n:
            log.info(f"Кэш содержит {len(cached)} игр — берём первые {n}")
            return cached[:n]

    log.info(f"Запрашиваем {n} игр из RAWG API …")
    games   = []
    page    = 1
    dates   = f"{YEARS_RANGE[0]}-01-01,{YEARS_RANGE[1]}-12-31"

    while len(games) < n:
        data = _rawg_get("/games", {
            "dates":               dates,
            "metacritic":          "40,100",    # только игры с Metacritic-рейтингом
            "ordering":            "-metacritic",
            "page_size":           PAGE_SIZE,
            "page":                page,
            "exclude_additions":   True,
        })
        if not data or not data.get("results"):
            log.warning("RAWG: пустой ответ, прекращаем.")
            break

        for g in data["results"]:
            games.append(_parse_rawg_game(g))

        log.info(f"  Страница {page}: +{len(data['results'])} игр  (итого {len(games)})")
        page += 1
        time.sleep(SLEEP_SEC)

        if not data.get("next"):
            break

    # Сохраняем кэш
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)
    log.info(f"Кэш сохранён: {CACHE_PATH}")

    return games[:n]


def _parse_rawg_game(g: dict) -> dict:
    """
    Извлекает нужные поля из RAWG-записи.

    Args:
        g: словарь RAWG API
    Returns:
        Упрощённый словарь
    """
    # Жанр — первый из списка
    genres    = g.get("genres")    or []
    platforms = g.get("platforms") or []

    genre_name = genres[0]["name"]    if genres    else "Unknown"
    platform_names = [p["platform"]["name"] for p in platforms]

    if len(platform_names) > 2:
        platform = "Multi-platform"
    elif any("PC" in p or "Windows" in p for p in platform_names):
        platform = "PC"
    elif any("PlayStation" in p for p in platform_names):
        platform = "PlayStation"
    elif any("Xbox" in p for p in platform_names):
        platform = "Xbox"
    else:
        platform = platform_names[0] if platform_names else "Unknown"

    # Год релиза
    released = g.get("released") or ""
    year = int(released[:4]) if len(released) >= 4 else None

    # Steam App ID (для дополнительных запросов)
    steam_id = None
    for store in (g.get("stores") or []):
        if store.get("store", {}).get("slug") == "steam":
            # URL вида: store.steampowered.com/app/12345/...
            url = store.get("url", "")
            parts = url.split("/app/")
            if len(parts) > 1:
                steam_id_str = parts[1].split("/")[0]
                if steam_id_str.isdigit():
                    steam_id = int(steam_id_str)
            break

    return {
        "title":          g.get("name", ""),
        "release_year":   year,
        "genre":          genre_name,
        "platform":       platform,
        "critic_score":   g.get("metacritic"),          # Metacritic score
        "user_score":     None,                          # заполним из Steam/SteamSpy
        "ratings_count":  g.get("ratings_count", 0),    # кол-во оценок на RAWG
        "rawg_rating":    g.get("rating"),               # оценка RAWG (0–5)
        "steam_app_id":   steam_id,
        "publisher":      None,
        "developer":      None,
        "budget_mln":     None,                          # нет в открытом доступе
        "revenue_mln":    None,
        "sales_mln":      None,
        "price_usd":      None,
        "peak_online":    None,
        "positive_pct":   None,                          # % позитивных отзывов Steam
        "tier":           None,
    }


# ══════════════════════════════════════════════
#  STEAM SPY API  (дополняет RAWG)
# ══════════════════════════════════════════════

def enrich_from_steamspy(games: list[dict]) -> list[dict]:
    """
    Обогащает данные из SteamSpy API:
      - owners (оценка числа владельцев → sales_mln)
      - positive / negative → positive_pct
      - price → price_usd

    SteamSpy не требует API-ключа, но ограничен ~1 запрос/сек.

    Args:
        games: список словарей (уже с steam_app_id)
    Returns:
        Обновлённый список
    """
    ids_with_steam = [(i, g) for i, g in enumerate(games) if g.get("steam_app_id")]
    log.info(f"SteamSpy: обогащаем {len(ids_with_steam)} игр со Steam App ID …")

    for idx, (i, g) in enumerate(ids_with_steam):
        try:
            resp = requests.get(
                STEAM_SPY,
                params={"request": "appdetails", "appid": g["steam_app_id"]},
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning(f"  SteamSpy [{g['steam_app_id']}]: {e}")
            time.sleep(SLEEP_SEC * 2)
            continue

        # owners — строка вида "1,000,000 .. 2,000,000"
        owners_str = data.get("owners", "0 .. 0")
        try:
            lo_str, hi_str = owners_str.replace(",", "").split(" .. ")
            owners_mid = (int(lo_str) + int(hi_str)) / 2
            games[i]["sales_mln"] = round(owners_mid / 1_000_000, 2)
        except ValueError:
            pass

        pos = data.get("positive", 0) or 0
        neg = data.get("negative", 0) or 0
        total = pos + neg
        if total > 0:
            games[i]["positive_pct"] = round(pos / total * 100, 1)
            # Пользовательская оценка: масштабируем % положительных → шкала 0–100
            games[i]["user_score"] = round(pos / total * 100, 1)

        price_raw = data.get("price") or data.get("initialprice")
        if price_raw:
            try:
                games[i]["price_usd"] = int(price_raw) / 100.0
            except (ValueError, TypeError):
                pass

        dev  = data.get("developer", "")
        pub  = data.get("publisher", "")
        if dev:
            games[i]["developer"] = dev
        if pub:
            games[i]["publisher"] = pub

        if (idx + 1) % 20 == 0:
            log.info(f"  … обработано {idx + 1}/{len(ids_with_steam)}")
        time.sleep(SLEEP_SEC)

    return games


# ══════════════════════════════════════════════
#  ПОСТОБРАБОТКА И СИНТЕТИЧЕСКОЕ ЗАПОЛНЕНИЕ
# ══════════════════════════════════════════════

def _fill_missing_with_synthetic(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Заполняет оставшиеся пропуски (бюджет, выручка, тир) синтетическими
    значениями, основанными на реальных распределениях по тиру и жанру.

    Args:
        df:   датафрейм с реальными данными
        seed: seed для воспроизводимости
    Returns:
        Заполненный датафрейм
    """
    rng = np.random.default_rng(seed)

    # ── Тир (AAA / AA / Indie) ──────────────────
    def assign_tier(row):
        if pd.notna(row.get("tier")):
            return row["tier"]
        cs = row.get("critic_score") or 0
        rc = row.get("ratings_count") or 0
        if cs >= 80 or rc > 50_000:
            return "AAA"
        elif cs >= 65 or rc > 10_000:
            return "AA"
        return "Indie"

    if "tier" not in df.columns or df["tier"].isna().all():
        df["tier"] = df.apply(assign_tier, axis=1)
    else:
        mask = df["tier"].isna()
        df.loc[mask, "tier"] = df[mask].apply(assign_tier, axis=1)

    # ── Бюджет (млн $) ──────────────────────────
    # Реальные диапазоны: AAA=50-300, AA=5-50, Indie=0.1-5
    BUDGET_PARAMS = {
        "AAA":   (120, 60),   # mean, std
        "AA":    (20,  15),
        "Indie": (1.5, 1.5),
    }
    mask = df["budget_mln"].isna()
    if mask.any():
        for tier, (mu, sigma) in BUDGET_PARAMS.items():
            t_mask = mask & (df["tier"] == tier)
            n = t_mask.sum()
            if n > 0:
                df.loc[t_mask, "budget_mln"] = np.clip(
                    rng.normal(mu, sigma, n), 0.1, 400
                ).round(2)

    # ── Выручка (млн $) ─────────────────────────
    REVENUE_MULT = {"AAA": (2.5, 1.2), "AA": (2.0, 1.0), "Indie": (3.0, 2.0)}
    mask = df["revenue_mln"].isna()
    if mask.any():
        for tier, (mult_mu, mult_sigma) in REVENUE_MULT.items():
            t_mask = mask & (df["tier"] == tier)
            n = t_mask.sum()
            if n > 0:
                mult = np.clip(rng.normal(mult_mu, mult_sigma, n), 0.1, 10)
                df.loc[t_mask, "revenue_mln"] = (
                    df.loc[t_mask, "budget_mln"] * mult
                ).round(2)

    # ── Продажи (млн копий) ─────────────────────
    mask = df["sales_mln"].isna()
    if mask.any():
        # грубая оценка: revenue / средняя цена / 2 (маркетинг)
        avg_price = df["price_usd"].median() if "price_usd" in df.columns else 29.99
        if pd.isna(avg_price):
            avg_price = 29.99
        df.loc[mask, "sales_mln"] = (
            df.loc[mask, "revenue_mln"] / max(avg_price, 1) / 2
        ).clip(0.01, 50).round(2)

    # ── Цена ────────────────────────────────────
    PRICE_MAP = {"AAA": 59.99, "AA": 39.99, "Indie": 14.99}
    mask = df["price_usd"].isna()
    if mask.any():
        df.loc[mask, "price_usd"] = df.loc[mask, "tier"].map(PRICE_MAP)

    # ── Пиковый онлайн ──────────────────────────
    mask = df["peak_online"].isna()
    if mask.any():
        # Коррелирует с продажами и оценкой игроков
        sales_med = df["sales_mln"].median() or 1
        df.loc[mask, "peak_online"] = (
            df.loc[mask, "sales_mln"] / sales_med * 10_000
            * np.clip(rng.exponential(1.5, mask.sum()), 0.1, 20)
        ).round(0).astype("Int64")

    # ── Оценка пользователей ────────────────────
    mask = df["user_score"].isna()
    if mask.any():
        # Связана с critic_score, но с шумом
        base = df.loc[mask, "critic_score"].fillna(65)
        noise = rng.normal(0, 12, mask.sum())
        df.loc[mask, "user_score"] = np.clip(base + noise, 10, 100).round(1)

    # ── positive_pct ────────────────────────────
    mask = df["positive_pct"].isna()
    if mask.any():
        base = df.loc[mask, "user_score"].fillna(65)
        df.loc[mask, "positive_pct"] = np.clip(base + rng.normal(0, 5, mask.sum()), 5, 100).round(1)

    # ── Разработчик / Издатель ──────────────────
    if "developer" not in df.columns:
        df["developer"] = None
    if "publisher" not in df.columns:
        df["publisher"] = None
    df["developer"].fillna("Неизвестно", inplace=True)
    df["publisher"].fillna("Неизвестно", inplace=True)

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит колонки к стандартным именам и типам.

    Args:
        df: датафрейм
    Returns:
        Нормализованный датафрейм
    """
    numeric_cols = [
        "release_year", "critic_score", "user_score",
        "budget_mln", "revenue_mln", "sales_mln",
        "price_usd", "peak_online", "positive_pct",
        "ratings_count", "rawg_rating",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["release_year"] = df["release_year"].astype("Int64")

    # Убираем строки без ключевых данных
    df = df.dropna(subset=["title", "critic_score"])
    df = df.reset_index(drop=True)
    return df


# ══════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════

def fetch_games_data(
    n: int = 200,
    force_refresh: bool = False,
    use_steamspy:  bool = True,
) -> pd.DataFrame:
    """
    Загружает данные о видеоиграх из RAWG API (+ SteamSpy) и возвращает DataFrame.

    Если RAWG_API_KEY не задан или недоступен, генерирует синтетические данные
    (аналогично предыдущей версии) и сохраняет в data/videogames.csv.

    Args:
        n:             количество игр (рекомендуется 200–500)
        force_refresh: игнорировать кэш и CSV, запросить заново
        use_steamspy:  обогащать данными SteamSpy (медленнее, но реалистичнее)

    Returns:
        pd.DataFrame с колонками:
            title, release_year, genre, platform, tier,
            critic_score, user_score, positive_pct,
            budget_mln, revenue_mln, sales_mln, price_usd,
            peak_online, developer, publisher,
            rawg_rating, ratings_count, steam_app_id
    """
    # ── 1. Проверяем готовый CSV ────────────────
    if not force_refresh and Path(CSV_PATH).exists():
        log.info(f"CSV уже существует: {CSV_PATH}  — загружаем.")
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        log.info(f"Загружено {len(df)} строк из CSV.")
        return df

    # ── 2. Пробуем реальный парсинг ─────────────
    if RAWG_API_KEY != "YOUR_API_KEY_HERE":
        log.info("═" * 50)
        log.info("Запускаем парсинг: RAWG API + SteamSpy")
        log.info("═" * 50)

        raw_games = fetch_rawg_games(n)

        if use_steamspy and raw_games:
            raw_games = enrich_from_steamspy(raw_games)

        if raw_games:
            df = pd.DataFrame(raw_games)
            df = _normalize_columns(df)
            df = _fill_missing_with_synthetic(df)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")
            log.info(f"✓ Данные сохранены: {CSV_PATH}  ({len(df)} строк)")
            return df
        else:
            log.warning("RAWG API вернул пустой результат, переходим к синтетике.")
    else:
        log.info("RAWG_API_KEY не задан → генерируем синтетические данные.")
        log.info("  Получите бесплатный ключ на https://rawg.io/apidocs")
        log.info("  и задайте: export RAWG_API_KEY=ваш_ключ")

    # ── 3. Резерв: синтетика ────────────────────
    log.info("Генерируем синтетический датасет …")
    df = _generate_synthetic(n)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    log.info(f"Синтетический датасет сохранён: {CSV_PATH}  ({len(df)} строк)")
    return df


# ══════════════════════════════════════════════
#  СИНТЕТИЧЕСКИЙ ГЕНЕРАТОР (резервный)
# ══════════════════════════════════════════════

def _generate_synthetic(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический датасет (идентичен исходному generate_dataset).

    Args:
        n:    количество строк
        seed: seed
    Returns:
        pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    genres     = ["Action", "RPG", "Shooter", "Adventure", "Sports",
                  "Strategy", "Simulation", "Horror"]
    developers = ["Studio A", "Studio B", "Studio C", "Studio D",
                  "Indie Dev", "AAA Corp", "Mid-size Games"]
    publishers = ["Publisher X", "Publisher Y", "Publisher Z", "Self-published"]
    platforms  = ["PC", "PlayStation", "Xbox", "Multi-platform"]
    tiers      = ["AAA", "AA", "Indie"]

    df = pd.DataFrame({
        "title":          [f"Game_{i:03d}" for i in range(n)],
        "release_year":   rng.integers(2018, 2026, size=n),
        "genre":          rng.choice(genres,     size=n),
        "developer":      rng.choice(developers, size=n),
        "publisher":      rng.choice(publishers, size=n),
        "platform":       rng.choice(platforms,  size=n),
        "tier":           rng.choice(tiers, size=n, p=[0.25, 0.30, 0.45]),
        "budget_mln":     np.clip(rng.exponential(60, n), 0.5, 400).round(2),
        "revenue_mln":    np.clip(rng.exponential(80, n), 0.1, 800).round(2),
        "critic_score":   np.clip(rng.normal(72, 14, n), 20, 100).round(1),
        "user_score":     np.clip(rng.normal(68, 18, n), 10, 100).round(1),
        "sales_mln":      np.clip(rng.exponential(3, n),  0.01, 30).round(2),
        "price_usd":      rng.choice([19.99, 29.99, 39.99, 49.99, 59.99, 69.99], size=n),
        "positive_pct":   np.clip(rng.normal(70, 15, n), 5, 100).round(1),
        "peak_online":    rng.integers(500, 200_000, size=n),
        "ratings_count":  rng.integers(100, 500_000, size=n),
        "rawg_rating":    np.clip(rng.normal(3.5, 0.8, n), 1, 5).round(2),
        "steam_app_id":   None,
    })

    # провальные ААА
    gap_idx = rng.choice(n, size=15, replace=False)
    df.loc[gap_idx, "user_score"] = np.clip(df.loc[gap_idx, "user_score"] - 40, 5, 100)

    # пропуски
    for col in ["budget_mln", "user_score", "developer", "peak_online"]:
        idx = rng.choice(n, size=int(n * 0.05), replace=False)
        df.loc[idx, col] = np.nan

    # дубликаты
    dup_idx = rng.choice(n, size=8, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


# ══════════════════════════════════════════════
#  БЫСТРЫЙ ТЕСТ
# ══════════════════════════════════════════════
if __name__ == "__main__":
    df = fetch_games_data(n=200)
    print(f"\nЗагружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(df.head(5).to_string())
    print(f"\nПропуски:\n{df.isnull().sum()[df.isnull().sum() > 0]}")