"""
Pipeline principal del predictor de dólar.
Orquesta: recolección → sentimiento → features → predicción.

Uso:
    python main.py --mode collect     # Solo recolectar datos
    python main.py --mode train       # Entrenar modelo
    python main.py --mode predict     # Predecir con modelo guardado
    python main.py --mode full        # Todo el pipeline
    python main.py --mode demo        # Demo sin API de Twitter
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime

# ─── Colores para la terminal ────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{CYAN}{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}{RESET}\n")


# ─── Modos ───────────────────────────────────────────────────────────────────

def mode_collect(days_back: int = 7, skip_twitter: bool = False):
    """Recolecta datos de X, La Nación y precios del dólar."""
    print_header("RECOLECCIÓN DE DATOS")

    # Precios del dólar (siempre disponible, sin API key)
    print(f"{YELLOW}[1/3] Precios del dólar (historial + hoy)...{RESET}")
    from collectors.dollar_collector import (
        append_current_price, get_all_types_today,
        fetch_historical_blue, save_historical
    )
    hist_df = fetch_historical_blue(days_back=365)
    if not hist_df.empty:
        save_historical(hist_df)
    get_all_types_today()
    append_current_price()

    # La Nación (RSS, sin API key)
    print(f"\n{YELLOW}[2/3] Noticias La Nación...{RESET}")
    from collectors.lanacion_collector import collect_and_save as collect_lanacion
    news_df = collect_lanacion(days_back=days_back)

    # Twitter/X (requiere API key)
    if not skip_twitter:
        print(f"\n{YELLOW}[3/3] Tweets sobre dólar...{RESET}")
        try:
            from collectors.twitter_collector import collect_and_save as collect_twitter
            tweets_df = collect_twitter(days_back=days_back)
        except ValueError as e:
            print(f"{RED}  ⚠  Twitter API no configurada: {e}{RESET}")
            print(f"     Configura X_BEARER_TOKEN en el archivo .env")
            tweets_df = pd.DataFrame()
    else:
        print(f"  Skipping Twitter (--no-twitter)")
        tweets_df = pd.DataFrame()

    return tweets_df, news_df


def mode_sentiment(tweets_df: pd.DataFrame = None, news_df: pd.DataFrame = None):
    """Aplica análisis de sentimiento a los textos recolectados."""
    print_header("ANÁLISIS DE SENTIMIENTO")

    from nlp.sentiment import analyze_dataframe, aggregate_daily_sentiment, compute_keyword_frequency
    from config import PROCESSED_DIR
    import os

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Sentimiento de tweets
    twitter_sentiment = pd.DataFrame()
    if tweets_df is not None and not tweets_df.empty:
        print(f"{YELLOW}Analizando {len(tweets_df)} tweets...{RESET}")
        tweets_df = analyze_dataframe(tweets_df, text_col="text")
        twitter_sentiment = aggregate_daily_sentiment(tweets_df, date_col="date")
        path = os.path.join(PROCESSED_DIR, "twitter_sentiment.csv")
        twitter_sentiment.to_csv(path, index=False)
        print(f"  Sentimiento Twitter guardado en {path}")

    # Frecuencia de keywords de La Nación
    news_features = pd.DataFrame()
    if news_df is not None and not news_df.empty:
        print(f"\n{YELLOW}Analizando {len(news_df)} artículos de La Nación...{RESET}")
        news_df = analyze_dataframe(news_df, text_col="title")
        news_sentiment = aggregate_daily_sentiment(news_df, date_col="date")
        news_kw = compute_keyword_frequency(news_df, text_col="title")

        if not news_kw.empty:
            news_features = news_sentiment.merge(news_kw, on="date", how="left")
        else:
            news_features = news_sentiment

        path = os.path.join(PROCESSED_DIR, "news_features.csv")
        news_features.to_csv(path, index=False)
        print(f"  Features de noticias guardadas en {path}")

    return twitter_sentiment, news_features


def mode_build_features(twitter_sentiment=None, news_features=None):
    """Construye la matriz de features para el modelo."""
    print_header("CONSTRUCCIÓN DE FEATURES")

    from features.feature_engineering import (
        load_dollar_history, build_feature_matrix, save_features
    )
    from config import PROCESSED_DIR

    # Cargar sentimiento si no se pasa directamente
    if twitter_sentiment is None:
        path = os.path.join(PROCESSED_DIR, "twitter_sentiment.csv")
        twitter_sentiment = pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else None

    if news_features is None:
        path = os.path.join(PROCESSED_DIR, "news_features.csv")
        news_features = pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else None

    dollar_df = load_dollar_history()
    print(f"  Días de precio disponibles: {len(dollar_df)}")

    df = build_feature_matrix(dollar_df, twitter_sentiment, news_features)
    print(f"  Features construidas: {len(df)} filas × {len(df.columns)} columnas")

    save_features(df)
    return df


def mode_train():
    """Entrena el modelo con las features disponibles."""
    from model import train_full_pipeline
    return train_full_pipeline()


def mode_predict():
    """Genera predicción con el modelo guardado."""
    print_header("PREDICCIÓN")

    from model import load_features, prepare_data, predict_tomorrow, load_model

    df = load_features()
    prediction = predict_tomorrow(df)

    direction_color = GREEN if prediction["recomendacion"] == "COMPRAR_DOLARES" else RED
    print(f"  Fecha:                  {prediction['date_predicted']}")
    print(f"  Precio actual:          ${prediction['current_price']:.2f}")
    print(f"  Dirección:              {direction_color}{prediction['predicted_direction']}{RESET} "
          f"(confianza: {prediction['confidence']:.1%})")
    print(f"  Retorno estimado 30d:  {prediction['predicted_return_30d_pct']:+.2f}%")
    print(f"  Precio estimado 30d:   ${prediction['predicted_price_30d']:.2f}")
    print(f"  Break-even PF 2%:       ${prediction['breakeven']:.2f}")
    print(f"  Recomendación:          {direction_color}{prediction['recomendacion']}{RESET}")

    return prediction


def mode_demo():
    """
    Demo completo sin requerir API de Twitter.
    Genera datos sintéticos para probar el pipeline.
    """
    print_header("MODO DEMO (sin Twitter API)")

    import numpy as np
    from config import RAW_DIR, PROCESSED_DIR
    import os

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Generar precios sintéticos del dólar blue (simulando tendencia real)
    print(f"{YELLOW}Generando datos sintéticos de precios...{RESET}")
    dates = pd.date_range(end=datetime.now().date(), periods=180, freq="D")
    np.random.seed(42)
    price = 1000.0
    prices = []
    for _ in dates:
        price *= (1 + np.random.normal(0.002, 0.015))  # drift + volatilidad
        prices.append(round(price, 2))

    dollar_df = pd.DataFrame({
        "date": dates,
        "type": "blue",
        "buy": [p * 0.98 for p in prices],
        "sell": prices,
    })
    dollar_path = os.path.join(RAW_DIR, "dollar_prices.csv")
    dollar_df.to_csv(dollar_path, index=False)
    print(f"  {len(dollar_df)} días de precios generados → {dollar_path}")

    # Generar sentimiento sintético de Twitter
    print(f"\n{YELLOW}Generando sentimiento sintético...{RESET}")
    sentiment_data = pd.DataFrame({
        "date": dates,
        "tweet_count": np.random.randint(50, 500, len(dates)),
        "avg_sentiment": np.random.uniform(-0.5, 0.5, len(dates)),
        "std_sentiment": np.random.uniform(0.1, 0.4, len(dates)),
        "pct_positive": np.random.uniform(0.2, 0.6, len(dates)),
        "pct_negative": np.random.uniform(0.2, 0.5, len(dates)),
        "pct_neutral": np.random.uniform(0.1, 0.3, len(dates)),
    })
    sentiment_path = os.path.join(PROCESSED_DIR, "twitter_sentiment.csv")
    sentiment_data.to_csv(sentiment_path, index=False)

    print(f"\n{YELLOW}Construyendo features...{RESET}")
    from features.feature_engineering import build_feature_matrix, save_features
    df = build_feature_matrix(dollar_df, sentiment_data, horizon=30)
    save_features(df)

    print(f"\n{YELLOW}Entrenando modelo...{RESET}")
    from model import train_full_pipeline
    prediction = train_full_pipeline()

    return prediction


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictor del dólar blue en Argentina"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "predict", "full", "demo"],
        default="demo",
        help="Modo de ejecución (default: demo)"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Días hacia atrás para recolectar (default: 7)"
    )
    parser.add_argument(
        "--no-twitter", action="store_true",
        help="Saltar recolección de Twitter"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "demo":
        mode_demo()

    elif args.mode == "collect":
        mode_collect(days_back=args.days, skip_twitter=args.no_twitter)

    elif args.mode == "train":
        mode_build_features()
        mode_train()

    elif args.mode == "predict":
        mode_predict()

    elif args.mode == "full":
        tweets_df, news_df = mode_collect(days_back=args.days, skip_twitter=args.no_twitter)
        twitter_sentiment, news_features = mode_sentiment(tweets_df, news_df)
        mode_build_features(twitter_sentiment, news_features)
        mode_train()
        mode_predict()
