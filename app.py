import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Credenciales OpenAI desde el .env de la raíz del repo .
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["tema"],
    template="""
    Explicame brevemente sobre el tema en forma sencilla: {tema}
    """
)

def explicar_tema(tema: str) -> str:
    prompt = prompt_template.format(tema=tema)
    respuesta = llm.invoke(prompt)  
    return respuesta.content


if __name__ == "__main__":  
    resultado = explicar_tema("La vida en la tierra")
    print(resultado)