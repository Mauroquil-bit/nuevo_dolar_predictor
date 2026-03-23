"""
Recolector de tweets sobre el dólar en Argentina.
Usa la API v2 de X (Twitter) con tweepy.
"""
import tweepy
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import X_BEARER_TOKEN, X_KEYWORDS, RAW_DIR


def get_client():
    if not X_BEARER_TOKEN:
        raise ValueError("X_BEARER_TOKEN no configurado en .env")
    return tweepy.Client(bearer_token=X_BEARER_TOKEN, wait_on_rate_limit=True)


def fetch_tweets(query_keywords: list[str], days_back: int = 7, max_per_query: int = 100) -> pd.DataFrame:
    """
    Descarga tweets de los últimos N días para cada keyword.
    El plan gratuito de X API v2 solo permite búsqueda de los últimos 7 días.
    """
    client = get_client()
    all_tweets = []

    end_time = datetime.utcnow() - timedelta(minutes=30)  # buffer requerido por la API
    start_time = end_time - timedelta(days=days_back)

    for keyword in query_keywords:
        query = f"{keyword} lang:es -is:retweet"
        print(f"  Buscando: '{keyword}'...")

        try:
            response = client.search_recent_tweets(
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=min(max_per_query, 100),
                tweet_fields=["created_at", "public_metrics", "lang"],
            )
            if response.data:
                for tweet in response.data:
                    all_tweets.append({
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": tweet.created_at,
                        "keyword": keyword,
                        "likes": tweet.public_metrics.get("like_count", 0),
                        "retweets": tweet.public_metrics.get("retweet_count", 0),
                        "replies": tweet.public_metrics.get("reply_count", 0),
                    })
            time.sleep(1)  # evitar rate limit

        except tweepy.TweepyException as e:
            print(f"  Error en '{keyword}': {e}")
            continue

    df = pd.DataFrame(all_tweets)
    if not df.empty:
        df = df.drop_duplicates(subset="id")
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date

    return df


def save_tweets(df: pd.DataFrame, filename: str = None):
    os.makedirs(RAW_DIR, exist_ok=True)
    if filename is None:
        filename = f"tweets_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Tweets guardados en {path} ({len(df)} registros)")
    return path


def collect_and_save(days_back: int = 7) -> pd.DataFrame:
    print("=== Recolectando tweets ===")
    df = fetch_tweets(X_KEYWORDS, days_back=days_back)
    if df.empty:
        print("No se encontraron tweets.")
        return df
    save_tweets(df)
    print(f"Total tweets recolectados: {len(df)}")
    return df


if __name__ == "__main__":
    collect_and_save(days_back=7)
