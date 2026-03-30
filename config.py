import os
from dotenv import load_dotenv

load_dotenv()

# X API
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# Palabras clave para buscar en X
X_KEYWORDS = [
    "dolar blue", "dolar hoy", "tipo de cambio",
    "devaluacion", "cepo cambiario", "reservas bcra",
    "dolar mep", "contado con liqui", "ccl",
    "inflacion argentina", "bcra", "economia argentina"
]

# Keywords negativos/positivos para filtrar ruido
NEGATIVE_SIGNALS = ["devaluacion", "corralito", "default", "cepo", "escasez reservas"]
POSITIVE_SIGNALS = ["acuerdo fmi", "reservas suben", "dolar baja", "estabilidad"]

# Fuentes RSS de economía (La Nación + otras fuentes sin paywall)
LANACION_RSS_FEEDS = [
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/category/economia/",
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/category/economia/finanzas-y-mercados/",
    "https://www.ambito.com/rss/pages/economia.xml",
    "https://www.cronista.com/files/rss/economia.xml",
    "https://www.infobae.com/feeds/rss/economia/",
]

# URL dólar (API pública argentina)
DOLAR_API_URL = "https://dolarapi.com/v1/dolares"

# Paths de datos
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models_saved"

# Modelo
TARGET_DOLAR = "blue"  # opciones: blue, mep, ccl, oficial
PREDICTION_HORIZON = 30  # días hacia adelante a predecir
LOOKBACK_DAYS = 7       # días de historia para features
