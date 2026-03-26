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
from config import PROCESSED_DIR, LOOKBACK_DAYS, TARGET_DOLAR


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

    # Retornos de corto plazo
    df["return_1d"] = price.pct_change(1)
    df["return_3d"] = price.pct_change(3)
    df["return_7d"] = price.pct_change(7)

    # Retornos de largo plazo
    df["return_14d"] = price.pct_change(14)
    df["return_30d"] = price.pct_change(30)

    # Volatilidad rolling
    df["volatility_7d"] = price.pct_change().rolling(7).std()
    df["volatility_14d"] = price.pct_change().rolling(14).std()

    # Volatilidad de largo plazo
    df["volatility_30d"] = price.pct_change().rolling(30).std()

    # Medias móviles
    df["ma_7d"] = price.rolling(7).mean()
    df["ma_14d"] = price.rolling(14).mean()
    df["ma_ratio"] = df["ma_7d"] / df["ma_14d"]

    # Medias móviles de largo plazo
    df["ma_30d"] = price.rolling(30).mean()
    df["ma_ratio_30_7"] = df["ma_30d"] / df["ma_7d"]

    # Posición del precio respecto a máximo/mínimo de 30 días
    df["max_30d"] = price.rolling(30).max()
    df["min_30d"] = price.rolling(30).min()
    df["pct_from_max_30d"] = (price - df["max_30d"]) / df["max_30d"]
    df["pct_from_min_30d"] = (price - df["min_30d"]) / df["min_30d"]

    # Días consecutivos sin movimiento mayor al 1%
    df["dias_quieto"] = (price.pct_change().abs() < 0.01).astype(int)
    df["racha_quieta"] = df["dias_quieto"].groupby(
        (df["dias_quieto"] != df["dias_quieto"].shift()).cumsum()
    ).cumcount() + 1
    df["racha_quieta"] = df["racha_quieta"] * df["dias_quieto"]

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


def add_target(df: pd.DataFrame, horizon: int = 30, pf_rate: float = 0.02) -> pd.DataFrame:
    """
    Target principal: ¿sube el dólar más del 2% en 30 días?
    - target_return_30d: retorno porcentual en 30 días
    - target_direction_30d: 1 si sube más que el plazo fijo, 0 si no
    - Mantiene también target_return (1 día) para compatibilidad
    """
    df = df.copy()
    # Target a N días (horizonte principal)
    df["target_price_30d"] = df["buy"].shift(-horizon)
    df["target_return_30d"] = df["target_price_30d"] / df["buy"] - 1
    df["target_direction_30d"] = (df["target_return_30d"] > pf_rate).astype(int)
    # Mantener target a 1 día para compatibilidad
    df["target_price"] = df["buy"].shift(-1)
    df["target_return"] = df["target_price"] / df["buy"] - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


def build_feature_matrix(
    dollar_df: pd.DataFrame,
    twitter_sentiment: pd.DataFrame = None,
    lanacion_news: pd.DataFrame = None,
    horizon: int = 30,
    pf_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Pipeline completo de construcción de features.
    """
    df = add_price_features(dollar_df)
    df = add_sentiment_features(df, twitter_sentiment)
    df = add_news_features(df, lanacion_news)
    df = add_target(df, horizon=horizon, pf_rate=pf_rate)

    # Eliminar filas con NaN en columnas críticas
    df = df.dropna(subset=["buy", "target_direction_30d"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna las columnas de features (excluye targets y metadatos)."""
    exclude = {"date", "type", "timestamp",
               "target_price", "target_return", "target_direction",
               "target_price_30d", "target_return_30d", "target_direction_30d",
               "buy", "sell", "dias_quieto", "max_30d", "min_30d", "ma_30d"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]


def save_features(df: pd.DataFrame, filename: str = None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if filename is None:
        filename = f"features_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Features guardadas en {path} ({len(df)} filas, {len(df.columns)} columnas)")
    return path
