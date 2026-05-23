"""
Modelo predictivo del precio del dólar.
Usa XGBoost para clasificación (sube/baja) y regresión (precio exacto).
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_absolute_percentage_error
)
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODELS_DIR, PROCESSED_DIR
from features.feature_engineering import get_feature_columns

# Half-life de los pesos exponenciales (en filas/días).
# Una muestra de hace 90 días pesa la mitad que la actual; 180 días = 1/4.
# El dataset tiene ~270 días, así que las muestras más viejas pesan ~0.13.
SAMPLE_WEIGHT_HALF_LIFE = 90


def load_features(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        # Usa el archivo más reciente
        files = sorted([
            f for f in os.listdir(PROCESSED_DIR) if f.startswith("features_")
        ])
        if not files:
            raise FileNotFoundError(f"No hay features en {PROCESSED_DIR}")
        filepath = os.path.join(PROCESSED_DIR, files[-1])
    print(f"Cargando features de: {filepath}")
    return pd.read_csv(filepath, parse_dates=["date"])


def prepare_data(df: pd.DataFrame):
    """Prepara X e y para entrenamiento, excluyendo filas sin target futuro."""
    feature_cols = get_feature_columns(df)
    train_df = df.dropna(subset=["target_return"])
    X = train_df[feature_cols].fillna(0)
    y_class = train_df["target_direction"].astype(int)
    y_reg = train_df["target_return"]
    return X, y_class, y_reg, feature_cols


def exponential_sample_weights(n: int, half_life: int = SAMPLE_WEIGHT_HALF_LIFE) -> np.ndarray:
    # Pesos exponenciales: la última fila pesa 1.0, una de hace `half_life` filas pesa 0.5.
    # Asume que X está ordenado cronológicamente (lo está: feature_engineering ordena por fecha).
    ages = np.arange(n)[::-1]  # [n-1, n-2, ..., 0]
    return 0.5 ** (ages / half_life)


def train_classifier(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Entrena clasificador XGBoost con validación temporal."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
    )

    # Validación cruzada con splits temporales (respeta el orden) — referencia sin pesos
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    print(f"CV Accuracy (sin pesos): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Entrenamiento final con sample weights exponenciales (recientes pesan más)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    sw_train = exponential_sample_weights(len(X_train))
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Baja", "Sube"]))

    return model, X_test, y_test, y_pred


def train_regressor(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Entrena regresor XGBoost para predecir retorno porcentual.
    Devuelve (modelo, X_test, y_test, y_pred, mae_return) — el MAE en retorno
    se usa luego para construir la banda de incertidumbre del precio estimado.
    """
    # Eliminar filas con NaN en el target (última fila no tiene precio futuro)
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        random_state=42,
    )

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    sw_train = exponential_sample_weights(len(X_train))
    model.fit(X_train, y_train, sample_weight=sw_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test + 1, y_pred + 1)
    print(f"Regressor MAE: {mae:.4f} | MAPE: {mape:.4f}")

    return model, X_test, y_test, y_pred, mae


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """Grafica importancia de features."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n)

    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", color="steelblue")
    plt.title(f"Top {top_n} Features más importantes")
    plt.xlabel("Importancia")
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {path}")


def plot_predictions(df_test: pd.DataFrame, y_test: pd.Series, y_pred_reg: np.ndarray):
    """Grafica predicciones vs valores reales."""
    plt.figure(figsize=(12, 5))
    plt.plot(df_test["date"].values if "date" in df_test.columns else range(len(y_test)),
             y_test.values, label="Real", color="blue", alpha=0.7)
    plt.plot(df_test["date"].values if "date" in df_test.columns else range(len(y_pred_reg)),
             y_pred_reg, label="Predicción", color="orange", alpha=0.7)
    plt.title("Retorno predicho vs real")
    plt.xlabel("Fecha")
    plt.ylabel("Retorno %")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {path}")


def save_model(model, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {path}")
    return path


def load_model(name: str):
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith(name) and f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError(f"No se encontró modelo {name} en {MODELS_DIR}")
    path = os.path.join(MODELS_DIR, files[-1])
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado: {path}")
    return model


def save_regressor_metadata(mae_return: float):
    """Persiste el MAE del regresor en JSON junto a los modelos.
    Se usa luego en predict_horizon para construir la banda ± de precio.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"regressor_metadata_{datetime.now().strftime('%Y%m%d')}.json")
    with open(path, "w") as f:
        json.dump({"mae_return": float(mae_return)}, f)
    print(f"Metadata del regresor guardada en {path}")
    return path


def load_regressor_metadata() -> dict:
    if not os.path.isdir(MODELS_DIR):
        return {}
    files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("regressor_metadata_") and f.endswith(".json")])
    if not files:
        return {}
    path = os.path.join(MODELS_DIR, files[-1])
    with open(path) as f:
        return json.load(f)


def predict_horizon(df: pd.DataFrame, classifier=None, regressor=None) -> dict:
    """
    Genera predicción para el horizonte configurado (30 días) usando la última fila disponible.
    """
    if classifier is None:
        classifier = load_model("classifier")
    if regressor is None:
        regressor = load_model("regressor")

    feature_cols = get_feature_columns(df)
    last_row = df[feature_cols].fillna(0).iloc[[-1]]

    direction = classifier.predict(last_row)[0]
    direction_proba = classifier.predict_proba(last_row)[0]
    predicted_return = regressor.predict(last_row)[0]

    current_price = df["buy"].iloc[-1]
    predicted_price = current_price * (1 + predicted_return)

    # Banda de incertidumbre: ± MAE del regresor (en retorno), expresado en pesos
    # sobre el precio actual. Si no hay metadata aún, banda = 0.
    metadata = load_regressor_metadata()
    mae_return = float(metadata.get("mae_return", 0.0))
    price_band = current_price * mae_return

    return {
        "date_predicted": datetime.now().date().isoformat(),
        "current_price": current_price,
        "predicted_direction": "SUBE" if direction == 1 else "BAJA",
        "confidence": float(max(direction_proba)),
        "predicted_return_pct": float(predicted_return * 100),
        "predicted_price": float(predicted_price),
        "predicted_price_low": float(predicted_price - price_band),
        "predicted_price_high": float(predicted_price + price_band),
        "mae_return_pct": float(mae_return * 100),
    }


def train_full_pipeline():
    """Pipeline completo de entrenamiento."""
    print("=" * 50)
    print("ENTRENAMIENTO DEL MODELO DÓLAR PREDICTOR")
    print("=" * 50)

    df = load_features()
    X, y_class, y_reg, feature_cols = prepare_data(df)

    print(f"\nDataset: {len(df)} días | {len(feature_cols)} features")
    print(f"Balance de clases: {y_class.value_counts().to_dict()}")

    print("\n--- Clasificador (sube/baja) ---")
    clf, X_test_c, y_test_c, y_pred_c = train_classifier(X, y_class)
    save_model(clf, "classifier")

    print("\n--- Regresor (retorno %) ---")
    reg, X_test_r, y_test_r, y_pred_r, mae_return = train_regressor(X, y_reg)
    save_model(reg, "regressor")
    save_regressor_metadata(mae_return)

    plot_feature_importance(clf, feature_cols)

    print("\n--- Estimación a 30 días ---")
    prediction = predict_horizon(df, clf, reg)
    print(f"\nPrecio actual:        ${prediction['current_price']:.2f}")
    print(f"Dirección predicha:   {prediction['predicted_direction']} "
          f"(confianza: {prediction['confidence']:.1%})")
    print(f"Retorno estimado 30d: {prediction['predicted_return_pct']:+.2f}%")
    print(f"Precio estimado 30d:  ${prediction['predicted_price']:.2f}")
    print(f"Banda ±MAE:           ${prediction['predicted_price_low']:.2f} – "
          f"${prediction['predicted_price_high']:.2f} (±{prediction['mae_return_pct']:.2f}%)")

    return prediction


if __name__ == "__main__":
    train_full_pipeline()
