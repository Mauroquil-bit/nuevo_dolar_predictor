# 🇦🇷 Dólar Predictor — ¿Plazo Fijo o Dólares?

[![GitHub Actions](https://github.com/Mauroquil-bit/nuevo_dolar_predictor/actions/workflows/daily_report.yml/badge.svg)](https://github.com/Mauroquil-bit/nuevo_dolar_predictor/actions/workflows/daily_report.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **La pregunta que se hacen miles de argentinos cada 30 días:**
> ¿Me conviene renovar el plazo fijo al 2% mensual, o me quedo en dólares?

📊 **[Ver el informe de hoy →](https://mauroquil-bit.github.io/nuevo_dolar_predictor/)**

---

## ¿Qué hace este proyecto?

Cada día hábil, de forma automática:

1. **Recolecta** el precio del dólar blue (compra/venta) via [dolarapi.com](https://dolarapi.com)
2. **Analiza** noticias económicas de La Nación (RSS) con NLP en español
3. **Reentrena** un modelo XGBoost con los datos históricos actualizados
4. **Predice** si el dólar blue va a subir más del 2% en los próximos 30 días
5. **Publica** el veredicto en una página web actualizada automáticamente

### El veredicto diario es simple:

| Situación | Recomendación |
|---|---|
| El modelo predice que el dólar **sube más del 2%** en 30 días | 💵 **QUEDARSE EN DÓLARES** |
| El modelo predice que el dólar **no supera el 2%** en 30 días | ✅ **RENOVAR PLAZO FIJO** |

> ⚠️ **Esto no es asesoramiento financiero profesional.** Es un experimento de machine learning educativo. Usalo como una señal más, nunca como única fuente de decisión.

---

## 📸 Vista del informe

El informe diario muestra:
- Precio actual del dólar blue (compra)
- Predicción a 30 días con nivel de confianza
- Break-even: cuánto debe subir el dólar para que no convenga el plazo fijo
- Tendencia de los últimos 7 días
- Recomendación clara: RENOVAR PLAZO FIJO o QUEDARSE EN DÓLARES

👉 [Ver informe en vivo](https://mauroquil-bit.github.io/nuevo_dolar_predictor/)

---

## 🏗️ Arquitectura del proyecto

```
nuevo_dolar_predictor/
├── .github/
│   └── workflows/
│       └── daily_report.yml    # Automatización diaria (GitHub Actions)
├── collectors/
│   ├── dollar_collector.py     # Precios via dolarapi.com (sin API key)
│   ├── lanacion_collector.py   # Noticias via RSS (sin paywall)
│   └── twitter_collector.py    # Tweets via X API v2 (opcional)
├── nlp/
│   └── sentiment.py            # Análisis de sentimiento en español
├── features/
│   └── feature_engineering.py  # Features técnicas + target a 30 días
├── model.py                    # XGBoost: clasificación + regresión
├── generate_report.py          # Genera index.html para GitHub Pages
├── main.py                     # Orquestador del pipeline
├── config.py                   # Configuración central
├── data/
│   ├── raw/                    # Precios históricos crudos
│   └── processed/              # Features procesadas
└── models_saved/               # Modelos entrenados (.pkl)
```

---

## ⚙️ Cómo funciona el modelo

### Features de entrada (variables que usa el modelo)

| Feature | Descripción |
|---|---|
| `return_1d`, `return_3d`, `return_7d`, `return_30d` | Retornos históricos del dólar blue |
| `volatility_7d`, `volatility_14d`, `volatility_30d` | Volatilidad rolling |
| `ma_7d`, `ma_14d`, `ma_30d`, `ma_ratio` | Medias móviles y su relación |
| `racha_quieta` | Días consecutivos sin movimiento > 1% |
| `pct_from_max_30d`, `pct_from_min_30d` | Distancia al máximo/mínimo del mes |
| `spread_pct` | Spread compra/venta |
| `buy_lag_1..N` | Precios anteriores (lags) |
| NLP features | Sentimiento de noticias de La Nación |

### Target (lo que predice)

**¿Va a subir el dólar blue más del 2% en los próximos 30 días?**
- `1` → Sí, supera el plazo fijo → Quedarse en dólares
- `0` → No, no supera el 2% → Renovar plazo fijo

### Pipeline de entrenamiento

```
Datos históricos → Features técnicas + NLP → XGBoost Classifier + Regressor
                                                      ↓
                                         Validación temporal (TimeSeriesSplit)
                                                      ↓
                                      Predicción: ¿supera 2% en 30 días?
```

---

## 🤖 Automatización diaria (GitHub Actions)

El workflow corre **de lunes a viernes a las 9:00 AM Argentina (12:00 UTC)**:

```yaml
- cron: "0 12 * * 1-5"
```

Pasos automáticos:
1. `python main.py --mode collect --no-twitter` — descarga precios del día
2. `python main.py --mode train` — reentrena el modelo
3. `python generate_report.py` — genera el HTML actualizado
4. `git commit & push` — publica en GitHub Pages

**Costo: $0** — todo corre en el plan gratuito de GitHub Actions (~3 min/día).

---

## 🚀 Instalación y uso local

### Requisitos
- Python 3.11+
- Git

### Instalación

```bash
git clone https://github.com/Mauroquil-bit/nuevo_dolar_predictor.git
cd nuevo_dolar_predictor
pip install -r requirements.txt
```

### Configuración (opcional — solo para Twitter/X)

```bash
cp .env.example .env
# Editar .env y agregar X_BEARER_TOKEN (opcional)
```

### Ejecutar

```bash
# Demo rápida con datos sintéticos (sin API key)
python main.py --mode demo

# Pipeline completo sin Twitter
python main.py --mode full --no-twitter

# Solo recolectar datos del día
python main.py --mode collect --no-twitter

# Solo entrenar con datos existentes
python main.py --mode train

# Solo predecir con modelo guardado
python main.py --mode predict

# Generar el informe HTML
python generate_report.py
```

---

## 📊 Rendimiento del modelo

| Métrica | Valor típico |
|---|---|
| Accuracy (clasificación) | 55–65% |
| MAE retorno 30d | ~2–4% |
| Validación | TimeSeriesSplit (5 folds) |
| Datos históricos | 300+ días de dólar blue |

> La accuracy de 55-65% puede parecer baja, pero es **mejor que azar** (50%) en un mercado altamente impredecible. El objetivo no es predecir perfectamente, sino tener una señal estadísticamente útil.

---

## 🔮 Roadmap / Ideas para contribuir

- [ ] Agregar fuentes RSS adicionales: **Infobae**, **Ámbito Financiero**, **El Cronista**
- [ ] Incorporar datos macroeconómicos: reservas BCRA, inflación mensual, tasa LELIQ
- [ ] Tasa del plazo fijo dinámica (hoy está hardcodeada al 2%)
- [ ] Backtesting histórico: ¿cuánto habrías ganado/perdido siguiendo las recomendaciones?
- [ ] Dashboard interactivo con Streamlit o Grafana
- [ ] Notificaciones por Telegram o WhatsApp cuando cambia la recomendación
- [ ] Soporte para otros tipos de dólar (MEP, CCL, Cripto)
- [ ] Fine-tuning de un LLM en español financiero argentino

---

## 🤝 Cómo contribuir

1. Hacé un fork del repositorio
2. Creá una rama: `git checkout -b feature/mi-mejora`
3. Commiteá tus cambios: `git commit -m "Agrego fuente Ambito"`
4. Hacé push: `git push origin feature/mi-mejora`
5. Abrí un Pull Request

**¿Sos financista y querés mejorar el modelo?** Abrí un Issue explicando tu idea. Las contribuciones de contexto del mercado argentino son muy bienvenidas.

---

## 📁 Fuentes de datos

| Fuente | Qué provee | API key |
|---|---|---|
| [dolarapi.com](https://dolarapi.com) | Precios dólar blue, oficial, MEP, CCL | No requerida |
| [Bluelytics](https://bluelytics.com.ar) | Histórico dólar blue | No requerida |
| La Nación RSS | Noticias económicas | No requerida |
| X (Twitter) API v2 | Sentimiento de mercado | Opcional |

---

## ⚠️ Disclaimer

Este proyecto es un **experimento educativo de machine learning**. No constituye asesoramiento financiero, bursátil ni cambiario. El mercado del dólar en Argentina está sujeto a decisiones políticas y regulatorias impredecibles que ningún modelo puede anticipar completamente.

**Siempre consultá con un asesor financiero profesional antes de tomar decisiones de inversión.**

---

## 📄 Licencia

MIT License — libre para usar, modificar y distribuir con atribución.

---

*Desarrollado con ❤️ para el ahorrista argentino.*
