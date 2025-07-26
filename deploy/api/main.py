import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
import tensorflow as tf
from tensorflow import keras
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL    = os.getenv("DB_DSN", "postgresql://ml:mlpass@db:5432/devices")
MODEL_PTH = os.getenv("MODEL_PATH", "/models/best_model.keras")
SEQ_LEN   = int(os.getenv("SEQ_LEN", "30"))
THRESH    = float(os.getenv("THRESH", "0.5"))

print(f"🔗 Conectando a DB: {DB_URL}")
print(f"📂 Cargando modelo LSTM: {MODEL_PTH}")

# Verificar si el archivo del modelo existe
if not os.path.exists(MODEL_PTH):
    raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PTH}")

engine = create_engine(DB_URL)

# Cargar modelo con manejo de errores mejorado
try:
    # Intentar cargar con custom_objects para compatibilidad
    model = keras.models.load_model(MODEL_PTH, compile=False)
    
    # Recompilar el modelo si es necesario
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo LSTM cargado correctamente")
    print(f"📊 Modelo input shape: {model.input_shape}")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("🔧 Intentando cargar sin compilación...")
    
    try:
        # Cargar sin compilación como alternativa
        import tensorflow.keras.utils as utils
        model = utils.get_custom_objects()
        model = keras.models.load_model(MODEL_PTH, compile=False)
        print("✅ Modelo cargado sin compilación")
    except Exception as e2:
        print(f"❌ Error crítico cargando modelo: {e2}")
        print("💡 Sugerencia: Regenera el modelo con la versión actual de Keras")
        # Crear un modelo dummy para que la API no falle completamente
        model = None

app = FastAPI(
    title="🔧 Device Failure Predictor API", 
    description="API de mantenimiento predictivo con LSTM",
    version="1.0"
)

def fetch_sequence(device_id: str, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    sql = """
    SELECT *
    FROM   device_features  -- Ajusta el nombre de tu tabla
    WHERE  device = %(dev)s
    ORDER BY timestamp DESC  -- Ajusta el nombre de tu columna de fecha
    LIMIT %(n)s
    """
    try:
        df = pd.read_sql(sql, engine, params={"dev": device_id, "n": seq_len})
        return df.sort_values("timestamp")  # Ascendente para LSTM
    except Exception as e:
        print(f"❌ Error consultando DB: {e}")
        raise HTTPException(500, f"Error de base de datos: {str(e)}")

def prepare_tensor(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if len(df) < SEQ_LEN:
        first_row = df.iloc[0:1]
        padding = pd.concat([first_row] * (SEQ_LEN - len(df)), ignore_index=True)
        df = pd.concat([padding, df], ignore_index=True)
    df = df.tail(SEQ_LEN)
    mat = df[feature_cols].astype(float).to_numpy()    
    return mat.reshape(1, SEQ_LEN, len(feature_cols))

class Prediction(BaseModel):
    device: str
    probability: float
    will_fail: bool
    confidence: float
    next_timestamp: int
    status: str

@app.get("/")
def root():
    return {
        "message": "🔧 API de Mantenimiento Predictivo", 
        "model": "LSTM",
        "version": "1.0",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_error",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__
    }

@app.get("/predict/{device_id}", response_model=Prediction)
def predict(device_id: str):
    """Predice la probabilidad de falla para el próximo día"""
    
    if model is None:
        raise HTTPException(503, "Modelo no disponible. Contacta al administrador.")
    
    # Obtener datos históricos
    df = fetch_sequence(device_id)
    if df.empty:
        raise HTTPException(404, f"Dispositivo '{device_id}' no encontrado")
    
    print(f"📊 Datos obtenidos para {device_id}: {len(df)} registros")
    
    # Preparar features (excluir columnas no numéricas)
    exclude_cols = {"device", "timestamp", "failure"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not feature_cols:
        raise HTTPException(400, "No se encontraron columnas de features")
    
    print(f"🔢 Features utilizados: {feature_cols}")
    
    # Preparar tensor y predecir
    try:
        tensor = prepare_tensor(df, feature_cols)
        print(f"🎯 Tensor shape: {tensor.shape}")
        
        # Predicción con LSTM
        y_pred = model.predict(tensor, verbose=0)
        prob = float(y_pred.ravel()[0])
        will_fail = prob >= THRESH
        confidence = abs(prob - 0.5) * 2  # Qué tan segura es la predicción
        
        # Próximo timestamp
        last_ts = df["timestamp"].iloc[-1]
        next_ts = int(last_ts + 1)
        
        status = "🚨 MANTENIMIENTO REQUERIDO" if will_fail else "✅ OK"
        
        print(f"✅ Predicción completada: {prob:.3f} ({'FALLA' if will_fail else 'OK'})")
        
        return Prediction(
            device=device_id,
            probability=prob,
            will_fail=will_fail,
            confidence=confidence,
            next_timestamp=next_ts,
            status=status
        )
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        raise HTTPException(500, f"Error en predicción: {str(e)}")

@app.get("/devices")
def list_devices():
    """Lista todos los dispositivos disponibles"""
    try:
        sql = "SELECT DISTINCT device FROM device_features LIMIT 100"
        df = pd.read_sql(sql, engine)
        return {"devices": df["device"].tolist()}
    except Exception as e:
        raise HTTPException(500, f"Error listando dispositivos: {str(e)}")

@app.get("/model-info")
def model_info():
    """Información sobre el modelo cargado"""
    if model is None:
        return {"error": "Modelo no cargado"}
    
    try:
        return {
            "model_loaded": True,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "layers": len(model.layers),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__
        }
    except Exception as e:
        return {"error": f"Error obteniendo info del modelo: {e}"}