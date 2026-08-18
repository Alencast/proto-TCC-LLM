import ollama

messages = [
    {
        "role": "system",
        "content": (
            "Você é um assistente de apoio para consultas sobre a LGPD. "
            "Converse em português de forma clara, objetiva e natural. "
            "Responda apenas com base nas informações disponíveis e "
            "não invente informações."
        )
    }
]

while True:
    print("\n" + "=" * 50)
    print("Digite sua pergunta")
    print("(ou digite 'sair' para encerrar)")
    print("=" * 50)

    text = input("\nVocê: ").strip()

    if text.lower() == "sair":
        print("\nAté mais!")
        break

    if not text:
        print("\nDigite uma pergunta.")
        continue

    messages.append(
        {
            "role": "user",
            "content": text
        }
    )

    print("\nAssistente: ", end="", flush=True)

    response = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        stream=True
    )

    resposta = ""

    for chunk in response:
        conteudo = chunk["message"]["content"]
        print(conteudo, end="", flush=True)
        resposta += conteudo

    print()

    messages.append(
        {
            "role": "assistant",
            "content": resposta
        }
    )