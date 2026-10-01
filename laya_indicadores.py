import json
import time
from pathlib import Path

import laya  # Importamos el módulo raíz para configurar los checkpoints
from laya import Router

# 2. Inicializar el Router (ahora sin argumentos inválidos)
router = Router()

# 2. Cargar explícitamente el checkpoint multilingüe oficial de Convai Innovations
# (Esto descargará mmBERT-base de 322M optimizado para más de 100 idiomas)
multilingual_agent = laya.load("convaiinnovations/laya-multilingual")

# 3. Adjuntar el agente al router bajo la etiqueta "multilingual"
# Esto fuerza a que todo el flujo de inferencia corra sobre este checkpoint
router.attach("multilingual", multilingual_agent)

body = "Mi nombre es walter y lo estoy llamando de Swiss Medical Seguros a través de un convenio con Banco Patagonia."

state = {
    "subject": "Identificar presentación de agente",
    "body": body,
}

# Mantener la escala 0-3 óptima para Laya
questions = {
    "presentation_score": {
        "type": "score",
        "instructions": "Evalúa el nivel de completitud de la presentación del agente en una escala del 0 al 3.",
        "criteria": [
            "No dice su nombre, ni que llama de Swiss Medical, ni menciona el convenio.", # Índice 0
            "Solo menciona un elemento (ej. solo dice su nombre, o solo la empresa).",      # Índice 1
            "Menciona dos de los tres elementos requeridos en la llamada.",                  # Índice 2
            "Cumple perfectamente: dice su nombre, que es de Swiss Medical y el convenio."  # Índice 3
        ]
    }
}

inicio = time.perf_counter()
# IMPORTANTE: Cambiado a .predict() para latencia mínima en tiempo real (<35ms)
result = router.predict(state, questions)
elapsed = time.perf_counter() - inicio

# Corregido: Usar la clave exacta de la pregunta
answer = result["answers"]["presentation_score"]

# Extraer el score continuo calculado por Laya (un float entre 0.0 y 3.0)
score_laya = answer["score"] 

# 2. Normalizar matemáticamente de la escala 0-3 a la escala 0-100
# Fórmula: (Score Actual / Score Máximo) * 100
score_100 = (score_laya / 3.0) * 100

# Lógica del semáforo
if score_100 >= 85:
    semaforo = "🟢 VERDE"
elif score_100 >= 45:
    semaforo = "🟡 AMARILLO"
else:
    semaforo = "🔴 ROJO"

print(f"[{semaforo}] - Score Laya original (0-3): {score_laya:.2f}")
print(f"Score Normalizado (0-100): {score_100:.1f}%")
print(f"Tiempo de respuesta total: {elapsed*1000:.1f} ms")

usage = result.get("usage") or {}
if usage:
    print(
        "usage:",
        {
            k: usage[k]
            for k in ("truncated", "state_tokens", "state_tokens_dropped", "windows")
            if k in usage
        },
    )
