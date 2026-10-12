from ollama import embed, chat
import math


# Carrega o conteúdo da LGPD
with open("documentos/lgpd.txt", "r", encoding="utf-8") as arquivo:
    texto = arquivo.read()


# Divide o documento em chunks de 500 caracteres
chunks = [
    texto[i:i + 500]
    for i in range(0, len(texto), 500)
]


# Cria embeddings para cada chunk
embeddings = []

for chunk in chunks:
    resposta = embed(
        model="nomic-embed-text",
        input=chunk
    )
    embeddings.append(resposta["embeddings"][0])


# Calcula a similaridade de cosseno entre dois vetores
def similaridade_cosseno(a, b):
    produto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))

    if norma_a == 0 or norma_b == 0:
        return 0

    return produto / (norma_a * norma_b)


# Recebe uma pergunta e recupera os chunks mais relevantes
def recuperar_chunks(pergunta, k=5):
    # Cria o embedding da pergunta
    embedding_pergunta = embed(
        model="nomic-embed-text",
        input=pergunta
    )["embeddings"][0]

    # Calcula a similaridade com cada chunk
    resultados = []

    for i, embedding in enumerate(embeddings):
        similaridade = similaridade_cosseno(
            embedding_pergunta,
            embedding
        )

        resultados.append((similaridade, i, chunks[i]))

    # Ordena do mais semelhante para o menos semelhante
    resultados.sort(reverse=True)

    return resultados[:k]


# Inicia o chat apenas quando rag.py é executado diretamente
if __name__ == "__main__":
    print(f"Documento dividido em {len(chunks)} chunks.")
    print("\nRAG da LGPD iniciado.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            print("Até mais!")
            break

        if not pergunta:
            continue

        # Recupera os 5 chunks mais relevantes
        resultados = recuperar_chunks(pergunta, k=5)

        # Monta o contexto que será enviado ao Qwen
        contexto = "\n\n".join(
            chunk for _, _, chunk in resultados
        )

        # Mostra os trechos recuperados
        print("\n--- Trechos recuperados ---\n")

        for posicao, (similaridade, indice, chunk) in enumerate(
            resultados, start=1
        ):
            print(
                f"[{posicao}] Chunk {indice} | "
                f"Similaridade: {similaridade:.4f}"
            )
            print(chunk)
            print()

        # Prompt enviado ao Qwen
        prompt = f"""
Responda à pergunta utilizando apenas as informações
presentes no contexto abaixo.

Não invente informações e não contradiga o contexto.
Caso o contexto não contenha informações suficientes,
informe que não há informações suficientes para responder.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}
"""

        # Gera a resposta usando o contexto recuperado
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

        print("Assistente:", response.message.content)
        print()