"""
Obtiene la tasa de plazo fijo 30 días desde estadísticasbcra.com
Requiere token gratuito: http://estadisticasbcra.com/api/registracion
Si no hay token configurado, usa el fallback de config.py
"""
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_pf_monthly_rate() -> float:
    """
    Retorna la tasa mensual de plazo fijo 30 días (ej: 0.0209 = 2.09%).
    Fuente: api.estadisticasbcra.com → variable tasa_depositos_30_dias (TNA%).
    Fallback: valor configurado en PF_MONTHLY_RATE_FALLBACK.
    """
    from config import ESTADISTICAS_BCRA_TOKEN, PF_MONTHLY_RATE_FALLBACK

    if not ESTADISTICAS_BCRA_TOKEN:
        print(f"  Tasa PF: usando fallback {PF_MONTHLY_RATE_FALLBACK*100:.2f}% mensual "
              f"(configurar ESTADISTICAS_BCRA_TOKEN para dato real)")
        return PF_MONTHLY_RATE_FALLBACK

    try:
        resp = requests.get(
            "https://api.estadisticasbcra.com/tasa_depositos_30_dias",
            headers={"Authorization": f"BEARER {ESTADISTICAS_BCRA_TOKEN}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # La API retorna lista de {"d": "YYYY-MM-DD", "v": TNA_porcentaje}
        latest_tna_pct = data[-1]["v"]  # ej: 25.10
        monthly_rate = latest_tna_pct / 100 / 12
        print(f"  Tasa PF: {latest_tna_pct:.2f}% TNA → {monthly_rate*100:.2f}% mensual")
        return monthly_rate
    except Exception as e:
        print(f"  Tasa PF: error al obtener dato real ({e}). Usando fallback {PF_MONTHLY_RATE_FALLBACK*100:.2f}%")
        return PF_MONTHLY_RATE_FALLBACK
