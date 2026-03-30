"""
Genera el reporte diario index.html con la predicción del modelo.
Uso: python generate_report.py
"""
import os
import sys
import pandas as pd
from datetime import date

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MESES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}


HISTORY_PATH = os.path.join("data", "predictions_history.csv")


def get_prediction():
    from features.feature_engineering import load_dollar_history, build_feature_matrix
    from model import predict_horizon
    from config import PROCESSED_DIR

    dollar_df = load_dollar_history()

    twitter_path = os.path.join(PROCESSED_DIR, "twitter_sentiment.csv")
    news_path = os.path.join(PROCESSED_DIR, "news_features.csv")
    twitter_sentiment = pd.read_csv(twitter_path, parse_dates=["date"]) if os.path.exists(twitter_path) else None
    news_features = pd.read_csv(news_path, parse_dates=["date"]) if os.path.exists(news_path) else None

    df = build_feature_matrix(dollar_df, twitter_sentiment, news_features)
    prediction = predict_horizon(df)
    return prediction, dollar_df


def save_prediction_to_history(prediction: dict):
    """Guarda la predicción del día en el historial para calcular precisión futura."""
    today = date.today().isoformat()
    row = pd.DataFrame([{
        "date": today,
        "current_price": prediction["current_price"],
        "predicted_direction": prediction["predicted_direction"],
        "predicted_price": prediction["predicted_price"],
        "confidence": prediction["confidence"],
    }])
    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        if today in history["date"].values:
            return  # Ya existe la entrada de hoy
        history = pd.concat([history, row], ignore_index=True)
    else:
        history = row
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)


def calculate_accuracy(dollar_df: pd.DataFrame) -> dict:
    """Calcula la precisión histórica comparando predicciones pasadas con resultados reales."""
    if not os.path.exists(HISTORY_PATH):
        return {"total": 0, "correct": 0, "accuracy": None}

    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    dollar_df = dollar_df.copy()
    dollar_df["date"] = pd.to_datetime(dollar_df["date"])

    correct = 0
    total = 0
    for _, row in history.iterrows():
        target_date = row["date"] + pd.Timedelta(days=30)
        actual = dollar_df[dollar_df["date"] >= target_date]
        if actual.empty:
            continue  # Todavía no pasaron 30 días
        actual_price = actual.iloc[0]["buy"]
        actual_direction = "SUBE" if actual_price > row["current_price"] else "BAJA"
        total += 1
        if actual_direction == row["predicted_direction"]:
            correct += 1

    accuracy = correct / total if total > 0 else None
    return {"total": total, "correct": correct, "accuracy": accuracy}


def fmt(p):
    return f"${p:,.0f}".replace(",", ".")


def build_price_rows(dollar_df, today):
    recent = dollar_df.tail(7).copy().reset_index(drop=True)
    rows = ""
    prev_buy = None
    for _, row in recent.iterrows():
        d = row["date"]
        buy = row["buy"]
        label = f"{d.day:02d}/{d.month:02d}"
        is_today = d.date() == today if hasattr(d, "date") else False

        if prev_buy is not None:
            diff = (buy - prev_buy) / prev_buy * 100
            if diff > 0.1:
                badge = f'<span class="text-xs text-rojo bg-red-50 px-2 py-0.5 rounded">▲ +{diff:.2f}%</span>'
            elif diff < -0.1:
                badge = f'<span class="text-xs text-verde bg-green-50 px-2 py-0.5 rounded">▼ {diff:.2f}%</span>'
            else:
                badge = '<span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">PLANO</span>'
        else:
            badge = '<span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">—</span>'

        today_tag = ' <span class="text-blue-400">(hoy)</span>' if is_today else ""
        price_class = "font-bold text-azul" if is_today else "font-bold"

        rows += f"""
          <div class="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
            <span class="text-gray-500">{label}{today_tag}</span>
            <div class="flex items-center gap-2">
              <span class="{price_class}">{fmt(buy)}</span>
              {badge}
            </div>
          </div>"""
        prev_buy = buy
    return rows


def render_html(prediction, dollar_df):
    from collectors.rates_collector import fetch_pf_monthly_rate

    today = date.today()
    dia = today.day
    mes = MESES[today.month]
    anio = today.year

    current_price = prediction["current_price"]
    direction = prediction["predicted_direction"]
    confidence = prediction["confidence"]
    predicted_price = prediction["predicted_price"]
    ret_pct = prediction["predicted_return_pct"]

    pf_monthly_rate = fetch_pf_monthly_rate()
    breakeven = current_price * (1 + pf_monthly_rate)

    accuracy_data = calculate_accuracy(dollar_df)
    if accuracy_data["accuracy"] is not None:
        precision_label = f"Precisión histórica: {accuracy_data['accuracy']:.0%} ({accuracy_data['correct']}/{accuracy_data['total']} predicciones)"
    else:
        precision_label = f"Precisión histórica: acumulando datos ({accuracy_data['total']} predicciones validadas)"

    # La recomendación se basa en si el precio estimado supera el break-even
    # Esto es más consistente que usar el clasificador, que puede contradecir al regresor
    recomendar_pf = predicted_price < breakeven
    display_direction = "SUBE" if ret_pct > 0 else "BAJA"
    dir_icon = "📈" if ret_pct > 0 else "📉"

    # Condicionales de estilo
    veredicto = "Hoy es buen momento para pasarte a plazo fijo" if recomendar_pf else "Quedate en dólares, no es momento de plazo fijo"
    veredicto_icon = "✅" if recomendar_pf else "⚠️"
    hero_class = "gradient-hero" if recomendar_pf else "gradient-hero-red"
    border_pred = "border-verde" if recomendar_pf else "border-rojo"
    text_pred = "text-verde" if recomendar_pf else "text-rojo"
    conclusion_bg = "bg-gradient-to-br from-azul to-celeste" if recomendar_pf else "bg-gradient-to-br from-red-900 to-red-700"
    accent = "text-green-300" if recomendar_pf else "text-yellow-300"
    alert_bg = "bg-green-50 border-green-200" if recomendar_pf else "bg-red-50 border-red-200"
    pulse_bg = "bg-verde" if recomendar_pf else "bg-rojo"
    pulse_class = "pulse-green" if recomendar_pf else "pulse-red"
    no_conviene_dolar = "no le gana al plazo fijo" if recomendar_pf else "le gana al plazo fijo"

    price_rows = build_price_rows(dollar_df, today)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>¿Renovar Plazo Fijo? — {dia} de {mes} de {anio}</title>

  <!-- Open Graph / Twitter Card -->
  <meta property="og:type" content="website" />
  <meta property="og:title" content="¿Renovar Plazo Fijo? — {dia} de {mes} {anio}" />
  <meta property="og:description" content="Blue compra: {fmt(current_price)} · Predicción: {direction} ({confidence:.0%} confianza) · {veredicto}" />
  <meta name="twitter:card" content="summary" />
  <meta name="twitter:title" content="¿Renovar Plazo Fijo? — {dia} de {mes} {anio}" />
  <meta name="twitter:description" content="Blue compra hoy: {fmt(current_price)} · El modelo predice {direction} con {confidence:.0%} de confianza → {veredicto}" />

  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      theme: {{
        extend: {{
          colors: {{
            azul: '#1e3a5f',
            celeste: '#2d7dd2',
            verde: '#0f9b57',
            rojo: '#d9382a',
            amarillo: '#f5a623',
          }}
        }}
      }}
    }}
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body {{ font-family: 'Inter', sans-serif; }}
    .gradient-hero {{ background: linear-gradient(135deg, #1e3a5f 0%, #2d7dd2 60%, #1a9e6e 100%); }}
    .gradient-hero-red {{ background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 60%, #92400e 100%); }}
    .card-shadow {{ box-shadow: 0 4px 24px rgba(0,0,0,0.08); }}
    .pulse-green {{ animation: pulseGreen 2s infinite; }}
    .pulse-red {{ animation: pulseRed 2s infinite; }}
    @keyframes pulseGreen {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(15,155,87,0.4); }}
      50% {{ box-shadow: 0 0 0 8px rgba(15,155,87,0); }}
    }}
    @keyframes pulseRed {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(217,56,42,0.4); }}
      50% {{ box-shadow: 0 0 0 8px rgba(217,56,42,0); }}
    }}
  </style>
</head>
<body class="bg-gray-50 text-gray-800">

  <header class="{hero_class} text-white py-14 px-6">
    <div class="max-w-4xl mx-auto">
      <div class="flex items-center gap-2 text-blue-200 text-sm mb-4 font-medium tracking-wide uppercase">
        <span>🇦🇷</span>
        <span>Análisis Cambiario · Dólar Blue Compra</span>
        <span class="mx-1">·</span>
        <span>{dia} de {mes} de {anio}</span>
      </div>
      <h1 class="text-4xl md:text-5xl font-extrabold leading-tight mb-4">
        ¿Renovar Plazo Fijo<br/>o Quedarse en Dólares?
      </h1>
      <p class="text-blue-100 text-lg max-w-2xl">
        Predicción diaria basada en {len(dollar_df)} días de precio del dólar blue comprador,
        sentimiento de noticias y modelo XGBoost.
      </p>
      <div class="mt-8 inline-flex items-center gap-3 bg-white/15 backdrop-blur border border-white/25 rounded-2xl px-6 py-4">
        <span class="text-3xl">{veredicto_icon}</span>
        <div>
          <div class="text-xs font-semibold uppercase tracking-widest text-blue-200">Recomendación de hoy</div>
          <div class="text-2xl font-extrabold text-white">{veredicto}</div>
          <div class="text-blue-200 text-sm">{precision_label} · XGBoost + NLP</div>
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-4xl mx-auto px-6 py-12 space-y-12">

    <!-- NÚMEROS DEL DÍA -->
    <section>
      <h2 class="text-2xl font-bold text-azul mb-6 flex items-center gap-2">
        <span class="text-3xl">💵</span> Los números de hoy
      </h2>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="bg-white rounded-2xl card-shadow p-5 border-l-4 border-celeste">
          <div class="text-xs text-gray-400 uppercase font-semibold tracking-wide mb-1">Blue compra hoy</div>
          <div class="text-3xl font-extrabold text-azul">{fmt(current_price)}</div>
          <div class="text-sm text-gray-500">Precio al que te compran los dólares</div>
        </div>
        <div class="bg-white rounded-2xl card-shadow p-5 border-l-4 {border_pred}">
          <div class="text-xs text-gray-400 uppercase font-semibold tracking-wide mb-1">Estimado 30 días</div>
          <div class="text-3xl font-extrabold {text_pred}">{fmt(predicted_price)}</div>
          <div class="text-sm text-gray-500">{dir_icon} {display_direction} en 30 días ({ret_pct:+.2f}%)</div>
        </div>
        <div class="bg-white rounded-2xl card-shadow p-5 border-l-4 border-amarillo">
          <div class="text-xs text-gray-400 uppercase font-semibold tracking-wide mb-1">Break-even (PF {pf_monthly_rate*100:.2f}%/mes)</div>
          <div class="text-3xl font-extrabold text-amarillo">{fmt(breakeven)}</div>
          <div class="text-sm text-gray-500">El blue debe superar esto en 30 días</div>
        </div>
      </div>
      <p class="mt-4 text-sm text-gray-600 {alert_bg} border rounded-xl px-4 py-3">
        {veredicto_icon} El plazo fijo rinde <strong>{pf_monthly_rate*100:.2f}% mensual</strong> → break-even en <strong>{fmt(breakeven)}</strong>.
        El modelo estima que el blue llegará a <strong>{fmt(predicted_price)}</strong> en 30 días,
        lo que <strong>{no_conviene_dolar}</strong>.
      </p>
    </section>

    <!-- TENDENCIA RECIENTE -->
    <section>
      <h2 class="text-2xl font-bold text-azul mb-6 flex items-center gap-2">
        <span class="text-3xl">📊</span> Últimos 7 días (precio compra)
      </h2>
      <div class="bg-white rounded-2xl card-shadow p-6">
        <div class="space-y-1 font-mono text-sm">
          {price_rows}
        </div>
        <div class="mt-4 flex items-center gap-2 text-sm {text_pred} font-medium">
          <span class="{pulse_class} w-2 h-2 {pulse_bg} rounded-full inline-block"></span>
          Estimación a 30 días: {display_direction} ({ret_pct:+.2f}%)
        </div>
      </div>
    </section>

    <!-- CONCLUSIÓN -->
    <section>
      <div class="{conclusion_bg} text-white rounded-2xl p-8">
        <h2 class="text-2xl font-extrabold mb-6 flex items-center gap-2">
          <span class="text-3xl">🏁</span> Conclusión
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div class="bg-white/15 rounded-xl p-5">
            <div class="text-3xl font-black {accent} mb-1">{veredicto_icon} {veredicto}</div>
            <div class="text-blue-100 text-sm">{dia} de {mes} de {anio}</div>
          </div>
          <div class="space-y-2">
            <div class="flex items-start gap-2 text-sm text-blue-100">
              <span class="{accent} font-bold mt-0.5">1.</span>
              <span>Blue compra actual: <strong class="text-white">{fmt(current_price)}</strong></span>
            </div>
            <div class="flex items-start gap-2 text-sm text-blue-100">
              <span class="{accent} font-bold mt-0.5">2.</span>
              <span>Precio estimado {display_direction} → <strong class="text-white">{fmt(predicted_price)}</strong> ({ret_pct:+.2f}%)</span>
            </div>
            <div class="flex items-start gap-2 text-sm text-blue-100">
              <span class="{accent} font-bold mt-0.5">3.</span>
              <span>Break-even del plazo fijo ({pf_monthly_rate*100:.2f}%/mes): <strong class="text-white">{fmt(breakeven)}</strong></span>
            </div>
            <div class="flex items-start gap-2 text-sm text-blue-100">
              <span class="{accent} font-bold mt-0.5">4.</span>
              <span>Precio estimado en 30 días: <strong class="text-white">{fmt(predicted_price)}</strong></span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- FOOTER -->
    <footer class="text-center text-xs text-gray-400 pt-4 pb-8 border-t border-gray-100">
      <p class="mb-1">⚠️ Análisis de datos automatizado. No constituye asesoramiento financiero profesional.</p>
      <p>Generado por <strong class="text-gray-500">Dólar Predictor</strong> ·
        XGBoost + Sentimiento NLP · Fuente: dolarapi.com, Bluelytics</p>
      <p class="mt-1 text-gray-300">Actualizado: {dia} de {mes} de {anio}</p>
    </footer>

  </main>
</body>
</html>"""


def main():
    print("Generando reporte diario...")
    prediction, dollar_df = get_prediction()

    direction = prediction["predicted_direction"]
    confidence = prediction["confidence"]
    current_price = prediction["current_price"]
    print(f"  Blue compra: {fmt(current_price)}")
    print(f"  Predicción 30d: {direction} ({confidence:.0%} confianza modelo)")
    ret_pct = prediction["predicted_return_pct"]
    predicted_price = prediction["predicted_price"]
    print(f"  Precio estimado 30d: ${predicted_price:.2f} ({ret_pct:+.2f}%)")

    save_prediction_to_history(prediction)

    html = render_html(prediction, dollar_df)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("OK - index.html generado correctamente")


if __name__ == "__main__":
    main()
