import json
import time
from pathlib import Path

from laya import Router

BODY_FILE = "baja2.json"

router = Router(preload=True)

with open(Path(__file__).parent / BODY_FILE, encoding="utf-8") as f:
    transcript = json.load(f)

# Texto legible: el JSON crudo gasta tokens en keys/timestamps y predict() trunca
# el inicio (~512 tokens). La resolución suele estar al final.
body = "\n".join(
    f"{item.get('channel', '?')}: {item.get('transcript', '').strip()}"
    for item in transcript
)

state = {
    "subject": "Resolución de llamada",
    "body": body,
}

questions = {
    "resolution": {
        "type": "choice",
        "instructions": "Como se resolvió la llamada?",
        "criteria": {
            "actualiza datos": "el cliente actualiza los datos personales o domicilio",
            "mantiene": "el cliente mantiene el plan",
            "cambia plan": "el cliente cambia el plan",
            "acepta porcentaje descuento": "el cliente acepta un porcentaje de descuento",
            "baja": "el agente acepta dar de baja o anular el producto",
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
