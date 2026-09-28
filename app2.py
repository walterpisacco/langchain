import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Credenciales OpenAI desde el .env de la raíz del repo .
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

#crear el prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

#crear cadena de conversacion
chain = prompt_template | llm


def ejecutar_chat() -> str:
    print("bIENVENIDO A MI CHATBOT. eSCRIBE 'salir' para salir.")
    chat_history = []
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "salir":
            break
        respuesta = chain.invoke({"input": user_input, "chat_history": chat_history})
        #actualizar el historial de conversacion
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=respuesta.content))

        print(f"Chatbot: {respuesta.content}")

ejecutar_chat()