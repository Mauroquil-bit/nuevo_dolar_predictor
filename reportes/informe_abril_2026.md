# Informe: ¿Plazo Fijo o Dólares en Abril 2026?

**Fecha de análisis:** 23 de marzo de 2026
**Modelo:** XGBoost + Sentimiento de noticias (La Nación, Ámbito)
**Fuente de precios:** Bluelytics API — 261 días de historial

---

## Resumen ejecutivo

> **Recomendación del modelo: PLAZO FIJO para abril.**
>
> El dólar blue está en zona de estabilidad. Para superar el rendimiento
> del plazo fijo necesitaría subir más del **2% en 30 días** — algo que
> el mercado no anticipó ni siquiera frente al riesgo de cierre del
> Estrecho de Ormuz.

---

## Los números

| Concepto | Valor |
|---|---|
| Inversión | $10.000 USD |
| Precio blue hoy | $1.425 |
| Equivalente en pesos | $14.250.000 |
| Ganancia plazo fijo 2% mensual | $285.000 pesos ≈ **$200 USD** |
| Umbral: el dólar debe superar | **$1.453,50** (+2%) para que no convenga |

---

## Qué dice el modelo

### Comportamiento histórico (últimos 261 días)

| Dirección | Días | % del tiempo |
|---|---|---|
| Baja o estable | 159 | **61%** |
| Sube | 102 | 39% |

### Predicción corto plazo
- **Dirección:** BAJA / ESTABLE — Confianza: **64%**
- **Tendencia reciente:** $1.430 → $1.425 → $1.425 (plano)

### Sentimiento de noticias (21–23 marzo)

| Día | Artículos | Sentimiento | % Negativo |
|---|---|---|---|
| 21/03 | 5 | -0.20 | 40% |
| 22/03 | 9 | -0.39 | 44% |
| 23/03 | 5 | -0.20 | 20% |

Titulares negativos detectados: inflación persistente, suba del riesgo país,
advertencia sobre reservas, deterioro del humor social.

---

## Variable geopolítica: El Estrecho de Ormuz

### ¿Qué es el Estrecho de Ormuz?

El Estrecho de Ormuz es un canal de apenas **55 km de ancho** ubicado entre
Irán y Omán. Es el cuello de botella energético más crítico del planeta:

- **20-21% del petróleo mundial** pasa por ahí (~17 millones de barriles/día)
- **25% del GNL global** también transita por ese estrecho
- Cualquier amenaza de cierre dispara el precio del Brent $15–$30 en días
- Genera presión inflacionaria global y fuga hacia activos de refugio (dólar, oro)

### El dato clave: no reaccionó el blue

Con el conflicto activo y el riesgo de cierre del Estrecho de Ormuz sobre la mesa,
**el dólar blue se mantuvo en $1.425**. Esto no es trivial.

### ¿Por qué Argentina resistió el shock?

| Factor | Explicación |
|---|---|
| Cepo cambiario | Contiene la demanda de dólares estructuralmente |
| Argentina = exportadora de energía | Vaca Muerta: el petróleo caro le beneficia |
| Acuerdo con el FMI | Credibilidad del programa económico |
| Alta tasa en pesos | El costo de dolarizarse es alto |
| Intervención del BCRA | Oferta sostenida en el mercado |

> El Estrecho de Ormuz es exactamente el tipo de shock externo que históricamente
> dispara el blue en Argentina. Si esta vez no lo movió, el ancla cambiaria
> es genuinamente fuerte. El modelo asigna **+4 puntos de estabilidad** a esta señal.

---

## Riesgos a monitorear en abril

| Riesgo | Probabilidad | Impacto en blue |
|---|---|---|
| Cierre efectivo del Estrecho de Ormuz | Baja-Media | +3% a +8% |
| Levantamiento/ajuste del cepo | Media | +5% a +15% |
| Desacuerdo con el FMI | Baja | +10% a +20% |
| Dato de inflación mayor al esperado | Alta | +1% a +3% |
| Tensión política electoral | Media | +3% a +8% |

---

## Conclusión final

**Plazo fijo en abril: SÍ conviene**, con base en:

1. El blue no reaccionó al riesgo de cierre del Estrecho de Ormuz → ancla fuerte
2. El modelo predice BAJA/ESTABLE con 64% de confianza
3. El 61% de los días en el último año el dólar bajó o se mantuvo
4. Ganancia segura de $200 USD en 30 días vs riesgo bajo de movimiento

**Acción:** Monitorear la primera semana de abril. Si hay novedades sobre
el cepo, el FMI, o escalada en el Estrecho de Ormuz, recalcular.

---

> ⚠️ Disclaimer: análisis de datos automatizado. No constituye asesoramiento financiero profesional.

*Generado por Dólar Predictor — `python main.py --mode full --no-twitter`*
