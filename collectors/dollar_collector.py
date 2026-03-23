"""
Recolector de precios históricos del dólar en Argentina.
Usa dolarapi.com (API pública, sin key requerida).
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOLAR_API_URL, RAW_DIR, TARGET_DOLAR

DOLAR_TYPES = {
    "oficial": "oficial",
    "blue": "blue",
    "mep": "bolsa",
    "ccl": "contadoconliqui",
    "cripto": "cripto",
    "mayorista": "mayorista",
}


def get_current_prices() -> dict:
    """Obtiene precios actuales de todos los tipos de dólar."""
    try:
        response = requests.get(DOLAR_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = {}
        for item in data:
            name = item.get("nombre", "").lower().replace(" ", "")
            prices[name] = {
                "buy": item.get("compra"),
                "sell": item.get("venta"),
                "timestamp": datetime.now().isoformat(),
            }
        return prices
    except Exception as e:
        print(f"Error obteniendo precios: {e}")
        return {}


def get_price_by_type(dolar_type: str = "blue") -> dict | None:
    """Obtiene precio de un tipo específico de dólar."""
    api_name = DOLAR_TYPES.get(dolar_type, dolar_type)
    url = f"https://dolarapi.com/v1/dolares/{api_name}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "type": dolar_type,
            "buy": data.get("compra"),
            "sell": data.get("venta"),
            "date": datetime.now().date().isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error obteniendo {dolar_type}: {e}")
        return None


def load_historical_prices(filepath: str = None) -> pd.DataFrame:
    """Carga precios históricos guardados localmente."""
    if filepath is None:
        filepath = os.path.join(RAW_DIR, "dollar_prices.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=["date", "type", "buy", "sell"])


def append_current_price(dolar_type: str = None) -> pd.DataFrame:
    """Agrega el precio actual al historial local."""
    if dolar_type is None:
        dolar_type = TARGET_DOLAR

    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, "dollar_prices.csv")

    df = load_historical_prices(filepath)
    new_price = get_price_by_type(dolar_type)

    if new_price:
        new_row = pd.DataFrame([new_price])
        new_row["date"] = pd.to_datetime(new_row["date"])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "type"], keep="last")
        df = df.sort_values("date")
        df.to_csv(filepath, index=False)
        print(f"Precio guardado: {dolar_type} = ${new_price['sell']} (venta)")

    return df


def get_all_types_today() -> pd.DataFrame:
    """Obtiene y guarda todos los tipos de cambio del día."""
    os.makedirs(RAW_DIR, exist_ok=True)
    records = []

    for dolar_type in DOLAR_TYPES:
        price = get_price_by_type(dolar_type)
        if price:
            records.append(price)
        time.sleep(0.5)

    df = pd.DataFrame(records)
    if not df.empty:
        today = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(RAW_DIR, f"all_dolares_{today}.csv")
        df.to_csv(filepath, index=False)
        print(f"Todos los tipos guardados en {filepath}")
        print(df[["type", "buy", "sell"]].to_string(index=False))
    return df


def fetch_historical_blue(days_back: int = 365) -> pd.DataFrame:
    """
    Descarga historial del dólar blue desde bluelytics.com.ar (API pública, sin key).
    Retorna DataFrame con columnas: date, type, buy, sell.
    """
    url = "https://api.bluelytics.com.ar/v2/evolution.json"
    print(f"Descargando historial desde bluelytics.com.ar...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        data = response.json()

        records = []
        cutoff = datetime.now() - timedelta(days=days_back)
        for item in data:
            date = pd.to_datetime(item.get("date"))
            if date < cutoff:
                continue
            source = item.get("source", "")
            if source == "Blue":
                records.append({
                    "date": date.date(),
                    "type": "blue",
                    "buy": item.get("value_buy"),
                    "sell": item.get("value_sell"),
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
            print(f"  {len(df)} dias descargados ({df['date'].min().date()} a {df['date'].max().date()})")
        return df

    except Exception as e:
        print(f"Error descargando historial: {e}")
        return pd.DataFrame()


def save_historical(df: pd.DataFrame):
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, "dollar_prices.csv")
    existing = load_historical_prices(filepath)
    combined = pd.concat([existing, df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date", "type"], keep="last").sort_values("date")
    combined.to_csv(filepath, index=False)
    print(f"Historial guardado en {filepath} ({len(combined)} registros)")
    return filepath


if __name__ == "__main__":
    print("=== Descargando historial dólar blue ===")
    df = fetch_historical_blue(days_back=365)
    if not df.empty:
        save_historical(df)
    print("\n=== Precios actuales ===")
    get_all_types_today()
