"""
Análisis de sentimiento financiero en español.
Implementación basada en keywords del dominio cambiario argentino.
No requiere torch ni modelos pesados — funciona en cualquier entorno.
"""
import pandas as pd
import numpy as np
import re
import os
import sys
import unicodedata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize(text: str) -> str:
    """Quita tildes y pasa a minúsculas para comparar keywords."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# ── Diccionario de sentimiento financiero argentino ──────────────────────────

# Todas las keywords en minúsculas SIN tildes (la normalización las quita antes de comparar)
POSITIVE = {
    # señal geopolitica: Estrecho de Ormuz activo SIN reacción del dólar = estabilidad fuerte
    "no impacto en el dolar": 4, "dolar no reacciono": 4, "dolar estable pese": 4,
    "sin impacto cambiario": 4, "mercado calmo": 3, "calma cambiaria": 3,
    "no afecta al dolar": 3, "estrecho de ormuz sin impacto": 4,
    "ormuz no afecta": 4, "blindado": 2, "guerra sin efecto": 3,

    # estabilidad / apreciación del peso
    "dolar baja": 3, "baja el dolar": 3, "dolar en baja": 3,
    "reservas suben": 3, "reservas crecen": 3, "reservas aumentan": 3,
    "acuerdo fmi": 3, "aprueba fmi": 3, "desembolso fmi": 3,
    "superavit": 2, "equilibrio fiscal": 2,
    "baja inflacion": 2, "inflacion baja": 2, "inflacion desacelera": 2,
    "estabilidad cambiaria": 2, "tipo de cambio estable": 2,
    "ingreso divisas": 2, "liquidacion campo": 2,
    "levantamiento cepo": 2, "apertura cambiaria": 2,
    "confianza": 1, "optimismo": 1, "recuperacion": 1, "crecimiento": 1,
    "inversion": 1, "bono sube": 1, "acciones suben": 1,
    "riesgo pais baja": 2, "baja riesgo pais": 2,
    "excedente": 1, "ahorro": 1, "baja la brecha": 2,
}

NEGATIVE = {
    # presión / depreciación
    "devaluacion": 3, "salto cambiario": 3,
    "dolar sube": 3, "sube el dolar": 3, "dolar al alza": 3, "dolar atrasado": 2,
    "brecha sube": 3, "brecha se amplia": 3, "brecha cambiaria": 2,
    "default": 3, "cese de pagos": 3,
    "reservas caen": 3, "reservas bajan": 3, "sin reservas": 3,
    "se va por la alcantarilla": 3, "alcantarilla": 2,
    "corrida cambiaria": 3, "corrida bancaria": 3,
    "cepo cambiario": 2, "restriccion cambiaria": 2,
    "inflacion sube": 2, "inflacion alta": 2, "inflacion persistente": 2,
    "inflacion de marzo": 1, "superar el 3": 1,
    "deficit fiscal": 2,
    "recesion": 2, "caida pbi": 2,
    "riesgo pais sube": 2, "riesgo pais": 1,
    "incertidumbre": 1, "volatilidad": 1, "tension": 1,
    "presion": 1, "crisis": 2, "alarma": 2, "alerta": 1,
    "fuga de capitales": 2, "morosidad": 2,
    "asfixia": 2, "encarece": 1, "impacto negativo": 2,
    "se deteriora": 2, "sin combustible": 1,
    "advertencia": 1, "preocupa": 2, "preocupacion": 2,
    "suba de tasas": 1, "tasas suben": 1,
}

NEUTRAL_OVERRIDE = [
    "dolar blue hoy", "precio dolar", "cotizacion dolar",
    "dolar a cuanto", "tipo de cambio hoy", "a cuanto cotiza",
    "precio en vivo",
]


def score_text(text: str) -> dict:
    """
    Calcula score de sentimiento de un texto financiero.
    Normaliza tildes antes de comparar keywords.
    """
    text_norm = normalize(text)

    # Primero chequear neutrales explícitos
    for neutral in NEUTRAL_OVERRIDE:
        if neutral in text_norm:
            return {"label": "NEU", "pos": 0.0, "neg": 0.0, "neu": 1.0, "score": 0.0}

    pos_score = 0.0
    neg_score = 0.0

    for phrase, weight in POSITIVE.items():
        if normalize(phrase) in text_norm:
            pos_score += weight

    for phrase, weight in NEGATIVE.items():
        if normalize(phrase) in text_norm:
            neg_score += weight

    total = pos_score + neg_score
    if total == 0:
        return {"label": "NEU", "pos": 0.0, "neg": 0.0, "neu": 1.0, "score": 0.0}

    pos_norm = pos_score / total
    neg_norm = neg_score / total
    score = pos_norm - neg_norm  # entre -1 y 1

    if score > 0.1:
        label = "POS"
    elif score < -0.1:
        label = "NEG"
    else:
        label = "NEU"

    return {
        "label": label,
        "pos": round(pos_norm, 4),
        "neg": round(neg_norm, 4),
        "neu": round(1 - abs(score), 4),
        "score": round(score, 4),
    }


def analyze_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Agrega columnas de sentimiento a un DataFrame."""
    df = df.copy()
    results = [score_text(str(t)) for t in df[text_col].fillna("")]
    sentiment_df = pd.DataFrame(results)
    df["sentiment_label"] = sentiment_df["label"].values
    df["sentiment_pos"]   = sentiment_df["pos"].values
    df["sentiment_neg"]   = sentiment_df["neg"].values
    df["sentiment_neu"]   = sentiment_df["neu"].values
    df["sentiment_score"] = sentiment_df["score"].values
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Agrega sentimientos por día."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    agg = df.groupby(date_col).agg(
        tweet_count    = ("sentiment_score", "count"),
        avg_sentiment  = ("sentiment_score", "mean"),
        std_sentiment  = ("sentiment_score", "std"),
        pct_positive   = ("sentiment_label", lambda x: (x == "POS").mean()),
        pct_negative   = ("sentiment_label", lambda x: (x == "NEG").mean()),
        pct_neutral    = ("sentiment_label", lambda x: (x == "NEU").mean()),
    ).reset_index()

    # Sentimiento ponderado por engagement si existe
    if "likes" in df.columns and "retweets" in df.columns:
        df["engagement"] = df["likes"].fillna(0) + df["retweets"].fillna(0) + 1
        df["weighted_s"] = df["sentiment_score"] * df["engagement"]
        weighted = (
            df.groupby(date_col)
            .apply(lambda g: g["weighted_s"].sum() / g["engagement"].sum())
            .reset_index(name="weighted_sentiment")
        )
        agg = agg.merge(weighted, on=date_col)

    agg["std_sentiment"] = agg["std_sentiment"].fillna(0)
    return agg


def compute_keyword_frequency(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Cuenta frecuencia de señales de alarma por día."""
    SIGNALS = {
        "sig_devaluacion":    ["devaluacion", "devaluación", "salto cambiario"],
        "sig_cepo":           ["cepo", "restriccion cambiaria"],
        "sig_reservas_bajas": ["sin reservas", "reservas caen", "reservas negativas"],
        "sig_acuerdo_fmi":    ["acuerdo fmi", "fmi aprueba", "desembolso fmi"],
        "sig_brecha":         ["brecha cambiaria", "brecha sube"],
        "sig_inflacion":      ["inflacion alta", "inflacion sube", "inflación alta"],
    # geopolitica: si hay shock del Estrecho de Ormuz Y el dólar no sube = estabilidad fuerte
    "sig_ormuz":          ["estrecho de ormuz", "cierre del estrecho", "bloqueo ormuz",
                           "iran bloquea", "petroleo sube", "suba del petroleo",
                           "conflicto medio oriente", "oriente medio", "tension iran"],
    "sig_geo_sin_impacto": ["no impacto en el dolar", "sin efecto cambiario", "dolar estable pese",
                            "mercado calmo", "calma cambiaria", "no reacciono el dolar"],
    }

    records = []
    for _, row in df.iterrows():
        text = normalize(str(row.get(text_col, "")))
        date = row.get("date")
        entry = {"date": date}
        for signal, keywords in SIGNALS.items():
            entry[signal] = int(any(kw in text for kw in keywords))
        records.append(entry)

    if not records:
        return pd.DataFrame()

    freq_df = pd.DataFrame(records)
    freq_df["date"] = pd.to_datetime(freq_df["date"])
    freq_df = freq_df.groupby("date").sum().reset_index()
    return freq_df
