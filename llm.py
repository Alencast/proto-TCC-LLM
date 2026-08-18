from ollama import chat

print("Qwen3 4B iniciado. Digite sua pergunta.")
print("Digite 'sair' para encerrar.\n")

while True:
    prompt = input("Você: ")

    if prompt.lower() == "sair":
        break

    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False
    )

    print(f"\nQwen: {response.message.content}\n")