import json
import time
from pathlib import Path

from laya import Router

body = "Mi nombre es walter y lo estoy llamando de Swiss Medical Seguros a través de un convenio con Banco Patagonia."

state = {
    "subject": "Identificar presentación de agente",
    "body": body,
}

questions = {
    "resolution": {
        "type": "choice",
        "instructions": "¿Dice quién llama, de donde llama y porqué llama?",
        "criteria": {
            "quien": "el agente dice su nombre",
            "de_donde": "el agente dice que llama de Swiss Medical",
            "porque": "el agente die que llama por un convenio con banco patagonia",
            "todos": "el agente dice su nombre, que llama de Swiss Medical y que es por un convenio con banco patagonia",
        }
    }
}

inicio = time.perf_counter()
# predict_long recorre ventanas solapadas; predict() solo ve el comienzo y corta el resto
result = router.predict_long(state, questions)
elapsed = time.perf_counter() - inicio

answer = result["answers"]["resolution"]
print(answer["choice"])
print(f"confidence: {answer.get('confidence')}")
print(f"Tiempo de respuesta total: {elapsed:.2f} s")
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
