import json
import time
from pathlib import Path

from laya import Router

BODY_FILE = "transcripcion-5086.json"

router = Router(preload=True)

with open(Path(__file__).parent / BODY_FILE, encoding="utf-8") as f:
    body = json.dumps(json.load(f), ensure_ascii=False)

state = {
    "subject": "Baja  de producto",
    "body": body,
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Como se resolvió la llamada de baja de producto?",
        "criteria": {
            "mantiene": "mantenimiento de plan",
            "actualiza": "actualización de plan",
            "baja 1": "baja por decision",
            "baja 2": "baja por desconocimiento",
            "baja 3": "baja por fallecimiento"
        }
    }
}

inicio = time.perf_counter()
result = router.predict(state, questions)
elapsed = time.perf_counter() - inicio

print(result["answers"]["department"]["choice"])
print(f"Tiempo de respuesta total: {elapsed:.2f} s")
