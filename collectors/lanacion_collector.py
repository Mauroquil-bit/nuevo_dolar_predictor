"""
Recolector de noticias económicas de La Nación via RSS.
No requiere API key, usa feeds RSS públicos.
"""
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import os
import sys
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANACION_RSS_FEEDS, RAW_DIR

ECONOMIA_KEYWORDS = [
    "dólar", "dolar", "tipo de cambio", "reservas", "bcra",
    "inflación", "inflacion", "cepo", "devaluación", "devaluacion",
    "economía", "economia", "fmi", "fiscal", "cambiario", "bonos"
]


def parse_rss_feed(url: str) -> list[dict]:
    """Parsea un feed RSS y retorna artículos como lista de dicts."""
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            try:
                pub_date = parsedate_to_datetime(entry.get("published", ""))
            except Exception:
                pub_date = datetime.now()

            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": pub_date,
                "date": pub_date.date() if pub_date else None,
                "source": "lanacion",
            })
    except Exception as e:
        print(f"Error parseando {url}: {e}")
    return articles


def filter_economic_articles(articles: list[dict]) -> list[dict]:
    """Filtra artículos que contienen keywords económicas."""
    filtered = []
    for article in articles:
        text = (article["title"] + " " + article["summary"]).lower()
        if any(kw in text for kw in ECONOMIA_KEYWORDS):
            filtered.append(article)
    return filtered


def scrape_article_text(url: str, timeout: int = 10) -> str:
    """
    Intenta extraer el texto del cuerpo de un artículo.
    La Nación tiene paywall, por lo que solo obtendremos el párrafo inicial.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, "html.parser")
        # Buscar párrafos del cuerpo del artículo
        paragraphs = soup.select("p.article-body__paragraph, p.sc-paragraph, article p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs[:5])
        return text[:1000]  # primeros 1000 chars
    except Exception:
        return ""


def collect_from_rss(days_back: int = 7) -> pd.DataFrame:
    """Recolecta noticias de todos los feeds RSS configurados."""
    cutoff = datetime.now() - timedelta(days=days_back)
    all_articles = []

    for feed_url in LANACION_RSS_FEEDS:
        print(f"  Feed: {feed_url}")
        articles = parse_rss_feed(feed_url)
        articles = filter_economic_articles(articles)
        # Filtrar por fecha
        articles = [a for a in articles if a["published"] and a["published"].replace(tzinfo=None) >= cutoff]
        all_articles.extend(articles)
        print(f"    {len(articles)} artículos relevantes encontrados")

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.drop_duplicates(subset="link")
        df["published"] = pd.to_datetime(df["published"], utc=True)
        df["date"] = df["published"].dt.date
    return df


def save_articles(df: pd.DataFrame, filename: str = None):
    os.makedirs(RAW_DIR, exist_ok=True)
    if filename is None:
        filename = f"lanacion_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Artículos guardados en {path} ({len(df)} registros)")
    return path


def collect_and_save(days_back: int = 7) -> pd.DataFrame:
    print("=== Recolectando noticias La Nación ===")
    df = collect_from_rss(days_back=days_back)
    if df.empty:
        print("No se encontraron artículos.")
        return df
    save_articles(df)
    return df


if __name__ == "__main__":
    collect_and_save(days_back=7)
