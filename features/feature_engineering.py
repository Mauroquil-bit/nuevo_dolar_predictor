"""
Construcción del dataset de features para el modelo predictivo.
Combina: precios históricos + sentimiento Twitter + señales La Nación.
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, LOOKBACK_DAYS, TARGET_DOLAR, PREDICTION_HORIZON


def load_dollar_history(filepath: str = None) -> pd.DataFrame:
    from config import RAW_DIR
    if filepath is None:
        filepath = os.path.join(RAW_DIR, "dollar_prices.csv")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df[df["type"] == TARGET_DOLAR].copy()
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features técnicas de serie temporal de precios.
    Usa el precio comprador (buy) ya que es el precio que recibe el usuario
    cuando vende dólares (ej: para pasarlos a plazo fijo).
    """
    df = df.copy().sort_values("date")
    price = df["buy"]

    # Retornos
    df["return_1d"] = price.pct_change(1)
    df["return_3d"] = price.pct_change(3)
    df["return_7d"] = price.pct_change(7)

    # Volatilidad rolling
    df["volatility_7d"] = price.pct_change().rolling(7).std()
    df["volatility_14d"] = price.pct_change().rolling(14).std()

    # Medias móviles
    df["ma_7d"] = price.rolling(7).mean()
    df["ma_14d"] = price.rolling(14).mean()
    df["ma_ratio"] = df["ma_7d"] / df["ma_14d"]

    # Spreads
    if "sell" in df.columns:
        df["spread"] = df["sell"] - df["buy"]
        df["spread_pct"] = df["spread"] / df["buy"]

    # Lags del precio comprador
    for lag in range(1, LOOKBACK_DAYS + 1):
        df[f"buy_lag_{lag}"] = price.shift(lag)

    return df


def add_sentiment_features(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge del DataFrame de precios con sentimiento diario.
    sentiment_df debe tener columna 'date'.
    """
    if sentiment_df is None or sentiment_df.empty:
        return df

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    df = df.merge(sentiment_df, on="date", how="left")

    # Sentimiento con lag (info del día anterior)
    sentiment_cols = [c for c in sentiment_df.columns if c != "date"]
    for col in sentiment_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    return df


def add_news_features(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de noticias de La Nación.
    news_df debe tener columnas de frecuencia de keywords por día.
    """
    if news_df is None or news_df.empty:
        return df

    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"])

    df = df.merge(news_df, on="date", how="left")
    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Agrega la variable target: variación porcentual del precio en N días.
    target = 1 si el precio sube, 0 si baja o se mantiene.
    """
    df = df.copy()
    df["target_price"] = df["buy"].shift(-horizon)
    df["target_return"] = df["target_price"] / df["buy"] - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def build_feature_matrix(
    dollar_df: pd.DataFrame,
    twitter_sentiment: pd.DataFrame = None,
    lanacion_news: pd.DataFrame = None,
    horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    """
    Pipeline completo de construcción de features.
    """
    df = add_price_features(dollar_df)
    df = add_sentiment_features(df, twitter_sentiment)
    df = add_news_features(df, lanacion_news)
    df = add_target(df, horizon=horizon)

    # Eliminar filas con NaN en columnas críticas
    df = df.dropna(subset=["buy", "target_direction"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna las columnas de features (excluye targets y metadatos)."""
    exclude = {"date", "type", "timestamp", "target_price", "target_return",
               "target_direction", "buy", "sell"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]


def save_features(df: pd.DataFrame, filename: str = None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if filename is None:
        filename = f"features_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Features guardadas en {path} ({len(df)} filas, {len(df.columns)} columnas)")
    return path
