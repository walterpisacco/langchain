import ast
import operator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Credenciales OpenAI desde el .env de la raíz del repo .
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_expr(node):
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("expresión no permitida")


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática. La entrada debe ser algo como '10 + 5'."""
    try:
        value = _eval_expr(ast.parse(expression, mode="eval"))
        return str(value)
    except Exception as exc:
        return f"No se pudo calcular: {exc}"


agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="Resuelve preguntas matemáticas usando la herramienta calculator.",
)


if __name__ == "__main__":
    resultado = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "¿Cuál es el resultado de la suma de 10 y 5?",
                }
            ]
        }
    )
    print(resultado["messages"][-1].content)
