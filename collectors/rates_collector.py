"""
Obtiene la tasa de plazo fijo 30 días desde estadísticasbcra.com
Requiere token gratuito: http://estadisticasbcra.com/api/registracion
Si no hay token configurado, usa el fallback de config.py
"""
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_bcra_monthly_rate() -> float:
    """Tasa del BCRA a 30 días (promedio del sistema bancario, no lo que recibe el ahorrista común)."""
    from config import ESTADISTICAS_BCRA_TOKEN, PF_MONTHLY_RATE_FALLBACK

    if not ESTADISTICAS_BCRA_TOKEN:
        return PF_MONTHLY_RATE_FALLBACK

    try:
        resp = requests.get(
            "https://api.estadisticasbcra.com/tasa_depositos_30_dias",
            headers={"Authorization": f"BEARER {ESTADISTICAS_BCRA_TOKEN}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        latest_tna_pct = data[-1]["v"]
        return latest_tna_pct / 100 / 365 * 30
    except Exception:
        return PF_MONTHLY_RATE_FALLBACK


def fetch_galicia_monthly_rate() -> float:
    """Tasa real Banco Galicia para ahorristas: 21% anual / 365 * 30 días."""
    from config import PF_GALICIA_MONTHLY_RATE
    return PF_GALICIA_MONTHLY_RATE


# Mantener nombre anterior para no romper código existente
def fetch_pf_monthly_rate() -> float:
    return fetch_bcra_monthly_rate()
