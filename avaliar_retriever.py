from rag import recuperar_chunks, chunks

K = 5

GROUND_TRUTH = {
    "Q01": {
        "pergunta": "O que é considerado dado pessoal pela LGPD?",
        "chunks_relevantes": [13],
    },
    "Q02": {
        "pergunta": "O que é considerado dado pessoal sensível?",
        "chunks_relevantes": [13],
    },
    "Q03": {
        "pergunta": "Quais princípios devem ser observados nas atividades de tratamento de dados pessoais?",
        "chunks_relevantes": [26, 27, 28, 29],
    },
    "Q04": {
        "pergunta": "Quais são os direitos assegurados ao titular dos dados pessoais?",
        "chunks_relevantes": [64, 65, 66, 67, 68, 69, 70, 71],
    },
    "Q05": {
        "pergunta": "Em quais situações o tratamento de dados pessoais pode ser realizado?",
        "chunks_relevantes": [30, 31, 32, 33],
    },
}


def precision_at_k(resultados, relevantes, k):
    recuperados = {indice for _, indice, _ in resultados[:k]}
    encontrados = recuperados.intersection(relevantes)
    return len(encontrados) / k if k > 0 else 0.0


def recall_at_k(resultados, relevantes, k):
    if not relevantes:
        return None

    recuperados = {indice for _, indice, _ in resultados[:k]}
    encontrados = recuperados.intersection(relevantes)

    return len(encontrados) / len(relevantes)


def reciprocal_rank(resultados, relevantes):
    for posicao, (_, indice, _) in enumerate(resultados, start=1):
        if indice in relevantes:
            return 1 / posicao

    return 0.0


def avaliar():
    resultados_metricas = []

    for codigo, dados in GROUND_TRUTH.items():
        relevantes = set(dados["chunks_relevantes"])

        resultados = recuperar_chunks(
            dados["pergunta"],
            k=K,
        )

        precision = precision_at_k(resultados, relevantes, K)
        recall = recall_at_k(resultados, relevantes, K)
        rr = reciprocal_rank(resultados, relevantes)

        resultados_metricas.append({
            "precision": precision,
            "recall": recall,
            "mrr": rr,
        })

        print(f"\n{'=' * 70}")
        print(f"{codigo}: {dados['pergunta']}")
        print(f"Precision@{K}: {precision:.4f}")
        print(f"Recall@{K}:    {recall:.4f}")
        print(f"RR:            {rr:.4f}")

        print("\nChunks recuperados:")

        for posicao, (similaridade, indice, chunk) in enumerate(
            resultados, start=1
        ):
            marcador = "*" if indice in relevantes else " "

            print(
                f"\n{marcador} Posição {posicao} | "
                f"Chunk {indice} | "
                f"Similaridade: {similaridade:.4f}"
            )

            if codigo in ("Q03", "Q05"):
                print("-" * 70)
                print(chunk)
                print("-" * 70)

        # Mostra os chunks originais próximos ao Art. 7º
        if codigo == "Q05":
            print("\n--- Chunks originais ao redor do Art. 7º ---")

            for indice in range(30, min(37, len(chunks))):
                print(f"\n--- Chunk {indice} ---")
                print(chunks[indice])

    if resultados_metricas:
        quantidade = len(resultados_metricas)

        media_precision = sum(
            r["precision"] for r in resultados_metricas
        ) / quantidade

        media_recall = sum(
            r["recall"] for r in resultados_metricas
        ) / quantidade

        mrr = sum(
            r["mrr"] for r in resultados_metricas
        ) / quantidade

        print(f"\n{'=' * 70}")
        print("MÉDIAS DO RETRIEVER")
        print(f"Precision@{K}: {media_precision:.4f}")
        print(f"Recall@{K}:    {media_recall:.4f}")
        print(f"MRR:           {mrr:.4f}")


if __name__ == "__main__":
    avaliar()