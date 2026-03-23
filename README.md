# Dólar Predictor 🇦🇷

Predice el precio del dólar blue en Argentina combinando:
- **X (Twitter)**: sentimiento del mercado en tiempo real
- **La Nación**: señales de noticias económicas (RSS, sin paywall)
- **dolarapi.com**: precios históricos (API pública, sin key)
- **XGBoost**: modelo de ML con features de series temporales + NLP

---

## Instalación

```bash
cd dolar-predictor
pip install -r requirements.txt
```

## Configuración

Copiar `.env.example` a `.env` y completar:

```bash
cp .env.example .env
```

La API de X (Twitter) es **opcional**. Sin ella, el modelo solo usa La Nación + precios.

Para obtener credenciales de X: https://developer.twitter.com/en/portal/dashboard

---

## Uso

### Demo rápida (sin API de Twitter)
```bash
python main.py --mode demo
```
Genera datos sintéticos, entrena y predice. Ideal para probar el pipeline.

### Pipeline completo
```bash
# 1. Recolectar datos (últimos 7 días)
python main.py --mode collect --days 7

# 2. Entrenar modelo
python main.py --mode train

# 3. Predecir mañana
python main.py --mode predict

# Todo de una vez
python main.py --mode full
```

### Sin Twitter
```bash
python main.py --mode full --no-twitter
```

---

## Estructura

```
dolar-predictor/
├── collectors/
│   ├── twitter_collector.py   # Tweets via X API v2
│   ├── lanacion_collector.py  # Noticias via RSS (sin key)
│   └── dollar_collector.py   # Precios via dolarapi.com
├── nlp/
│   └── sentiment.py           # Análisis de sentimiento en español
├── features/
│   └── feature_engineering.py # Construcción de features
├── model.py                   # Entrenamiento y predicción XGBoost
├── main.py                    # Orquestador del pipeline
├── config.py                  # Configuración central
└── data/
    ├── raw/                   # Datos crudos
    └── processed/             # Features procesadas
```

---

## Salida del modelo

```
Precio actual:      $1,250.00
Dirección:          SUBE (confianza: 67.3%)
Retorno estimado:  +1.82%
Precio estimado:   $1,272.75
```

---

## Limitaciones

- El dólar en Argentina responde a decisiones políticas impredecibles
- El X API gratuito solo permite búsquedas de los últimos 7 días
- La Nación tiene paywall, se usa solo título + resumen del RSS
- Accuracy típica: 55-65% (mejor que azar, pero no fiable para trading)

---

## Extender el proyecto

- Agregar **Infobae**, **Ámbito**, **El Cronista** como fuentes RSS adicionales
- Incorporar datos macroeconómicos (reservas BCRA, inflación mensual)
- Fine-tuning de un LLM en español financiero
- Dashboard con Streamlit para visualización en tiempo real
