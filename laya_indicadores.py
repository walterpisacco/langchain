import time

import laya
from laya import Router

router = Router()
# Checkpoint mmBERT (100+ idiomas). El texto va en español; las preguntas, en inglés:
# el encoder multilingüe lee el turno, pero el cabezal de decisión se entrenó con
# instrucciones en inglés. En español las mismas preguntas casi no detectan el nombre.
router.attach("multilingual", laya.load("convaiinnovations/laya-multilingual"))

body = "Mi nombre es walter y lo estoy llamando de Swiss Medical Seguros a través de un convenio con Banco Patagonia."

state = {
    "subject": "Identificar presentación de agente",
    "body": body,
}

# Una rúbrica 0-3 no sirve aquí. Laya puntúa cada nivel por parecido con el texto,
# y el nivel 0 ("No dice su nombre, ni Swiss Medical, ni el convenio") repite las
# mismas palabras que el turno. En este ejemplo ese nivel se lleva ~43% y el nivel
# 1 otro ~42%, así que el score esperado cae a ~0.8 aunque la presentación esté completa.
# Tres preguntas sí/no, sin repetir la entidad en la opción negativa, sí se separan.
questions = {
    "nombre": {
        "type": "noul",
        "instructions": "Does the speaker state their own first name in `body`?",
    },
    "empresa": {
        "type": "noul",
        "instructions": "Does the speaker say they are calling from Swiss Medical Seguros in `body`?",
    },
    "convenio": {
        "type": "noul",
        "instructions": "Does the speaker mention a convenio or agreement with Banco Patagonia in `body`?",
    },
}

inicio = time.perf_counter()
result = router.predict(state, questions, lang="es")
elapsed = time.perf_counter() - inicio

partes = {qid: result["answers"][qid]["noul"] for qid in questions}
score_100 = 100.0 * sum(partes.values()) / len(partes)

if score_100 >= 85:
    semaforo = "🟢 VERDE"
elif score_100 >= 45:
    semaforo = "🟡 AMARILLO"
else:
    semaforo = "🔴 ROJO"

routing = result.get("routing") or {}
print(f"[{semaforo}] - Score (0-100): {score_100:.1f}%")
for qid, valor in partes.items():
    print(f"  {qid}: {valor:.2f}")
print(f"Modelo: {routing.get('model')} | idioma: es")
print(f"Tiempo de respuesta total: {elapsed * 1000:.1f} ms")

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
