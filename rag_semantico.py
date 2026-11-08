
import re
import math
import unicodedata
from collections import Counter

import numpy as np

from ollama import embed, chat


# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen3:4b"

FILE_PATH = "documentos/lgpd.txt"

SIMILARITY_THRESHOLD_PERCENTILE = 90
MAX_CHUNK_CHARS = 2500

TOP_K = 5
TEMPERATURE = 0.1

# Pesos da recuperação híbrida.
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45

# Parâmetros do BM25.
BM25_K1 = 1.5
BM25_B = 0.75

# Reconhece artigos somente no início de uma linha.
# Evita confundir referências a outros artigos com novos artigos.
ARTICLE_PATTERN = re.compile(
    r"^[ \t]*(Art\.\s*\d+(?:[º°o])?(?:-[A-Za-z])?)(?=[ \t.:])",
    re.IGNORECASE | re.MULTILINE
)

# Prefixos específicos do Nomic.
TASK_PREFIXES = {
    "document": "search_document: ",
    "query": "search_query: ",
    "clustering": "clustering: ",
}

# Palavras sem relevância significativa para a busca lexical.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre",
    "e", "ou", "que", "se", "ao", "aos",
    "é", "ser", "são", "foi", "será",
    "qual", "quais", "como", "quando", "onde",
    "deve", "devem", "pode", "podem",
    "seu", "sua", "seus", "suas",
    "este", "esta", "esse", "essa",
}


# ==========================================
# 2. GERAÇÃO DE EMBEDDINGS
# ==========================================

def get_embedding(text: str, task: str = "document") -> list[float]:
    """Gera embeddings utilizando o prefixo adequado à tarefa."""

    if task not in TASK_PREFIXES:
        raise ValueError(f"Tarefa de embedding inválida: {task}")

    response = embed(
        model=EMBEDDING_MODEL,
        input=TASK_PREFIXES[task] + text
    )

    return response["embeddings"][0]


# ==========================================
# 3. SIMILARIDADE DE COSSENO
# ==========================================

def cosine_similarity(a, b):
    """Calcula a similaridade de cosseno entre dois vetores."""

    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norma_a * norma_b))


# ==========================================
# 4. NORMALIZAÇÃO E TOKENIZAÇÃO
# ==========================================

def normalize_text(text: str) -> str:
    """Remove acentos e normaliza o texto para a busca lexical."""

    normalized = unicodedata.normalize("NFD", text.lower())

    return "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    """Extrai termos relevantes para a recuperação lexical."""

    normalized = normalize_text(text)

    tokens = re.findall(r"[a-z0-9]+", normalized)

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normaliza pontuações para o intervalo entre 0 e 1."""

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [0.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


# ==========================================
# 5. DIVISÃO DE BLOCOS MUITO GRANDES
# ==========================================

def split_oversized_paragraph(
    paragraph: str,
    max_chars: int
):
    """
    Divide blocos grandes priorizando pontuação e quebras de linha.
    Como alternativa, utiliza separação por palavras.
    """

    paragraph = paragraph.strip()

    if len(paragraph) <= max_chars:
        return [paragraph]

    fragments = re.split(
        r"(?<=[.;:!?])\s+|\n+",
        paragraph
    )

    result = []
    current = ""

    for fragment in fragments:
        fragment = fragment.strip()

        if not fragment:
            continue

        if len(fragment) > max_chars:

            if current:
                result.append(current)
                current = ""

            words = fragment.split()
            word_chunk = ""

            for word in words:

                if len(word) > max_chars:

                    if word_chunk:
                        result.append(word_chunk)
                        word_chunk = ""

                    for i in range(0, len(word), max_chars):
                        result.append(word[i:i + max_chars])

                    continue

                candidate = (
                    f"{word_chunk} {word}"
                    if word_chunk
                    else word
                )

                if len(candidate) > max_chars:

                    if word_chunk:
                        result.append(word_chunk)

                    word_chunk = word

                else:
                    word_chunk = candidate

            if word_chunk:
                result.append(word_chunk)

            continue

        candidate = (
            f"{current} {fragment}"
            if current
            else fragment
        )

        if len(candidate) > max_chars:

            if current:
                result.append(current)

            current = fragment

        else:
            current = candidate

    if current:
        result.append(current)

    return result


# ==========================================
# 6. SEPARAÇÃO E IDENTIFICAÇÃO DOS ARTIGOS
# ==========================================

def split_document_sections(text: str):
    """
    Separa o documento em seções respeitando o início
    dos artigos da legislação.
    """

    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        return [("Documento", text)]

    sections = []

    preamble = text[:matches[0].start()].strip()

    if preamble:
        sections.append(("Preâmbulo", preamble))

    for i, match in enumerate(matches):

        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        article_label = match.group(1).strip()
        article_text = text[start:end].strip()

        if article_text:
            sections.append((article_label, article_text))

    return sections


# ==========================================
# 7. CHUNKING HÍBRIDO
# ==========================================

def semantic_chunking(
    text: str,
    percentile_threshold: float = SIMILARITY_THRESHOLD_PERCENTILE
):
    """
    Segmenta o documento combinando:

    1. Fronteiras estruturais entre artigos.
    2. Mudanças de similaridade semântica.
    3. Limite máximo de caracteres.

    Os chunks de continuação preservam a identificação
    do artigo ao qual pertencem.
    """

    if not 0 <= percentile_threshold <= 100:
        raise ValueError(
            "O percentil deve estar entre 0 e 100."
        )

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    sections = split_document_sections(text)

    if not sections:
        return []

    print(f"Artigos/seções identificados: {len(sections)}")
    print("Preparando blocos para análise semântica...")

    content_limit = MAX_CHUNK_CHARS - 100

    prepared_sections = []
    all_distances = []

    for article_label, section_text in sections:

        raw_paragraphs = re.split(
            r"\n\s*\n+",
            section_text
        )

        paragraphs = []

        for paragraph in raw_paragraphs:

            paragraph = paragraph.strip()

            if paragraph and len(paragraph) > 5:

                paragraphs.extend(
                    split_oversized_paragraph(
                        paragraph,
                        content_limit
                    )
                )

        if not paragraphs:
            continue

        print(
            f"Processando {article_label}: "
            f"{len(paragraphs)} blocos"
        )

        paragraph_embeddings = [
            get_embedding(paragraph, task="clustering")
            for paragraph in paragraphs
        ]

        section_distances = []

        for i in range(len(paragraph_embeddings) - 1):

            similarity = cosine_similarity(
                paragraph_embeddings[i],
                paragraph_embeddings[i + 1]
            )

            distance = 1.0 - similarity

            section_distances.append(distance)
            all_distances.append(distance)

        prepared_sections.append({
            "article": article_label,
            "paragraphs": paragraphs,
            "distances": section_distances,
        })

    if not prepared_sections:
        return []

    if all_distances:

        threshold = np.percentile(
            all_distances,
            percentile_threshold
        )

    else:
        threshold = float("inf")

    print(
        f"\nLimiar semântico "
        f"(percentil {percentile_threshold}): "
        f"{threshold:.4f}"
    )

    chunks = []

    for section in prepared_sections:

        article_label = section["article"]
        paragraphs = section["paragraphs"]
        distances = section["distances"]

        current_chunk = [paragraphs[0]]
        current_size = len(paragraphs[0])

        first_chunk_of_article = True

        for i, distance in enumerate(distances):

            next_paragraph = paragraphs[i + 1]

            # Verifica o limite de caracteres.
            exceeds_max_size = (
                current_size
                + 2
                + len(next_paragraph)
                > content_limit
            )

            # Verifica mudança semântica.
            semantic_break = distance >= threshold

            if exceeds_max_size or semantic_break:

                body = "\n\n".join(current_chunk)

                if not first_chunk_of_article:
                    body = (
                        f"[Continuação de {article_label}]\n"
                        f"{body}"
                    )

                chunks.append(body)

                current_chunk = [next_paragraph]
                current_size = len(next_paragraph)

                first_chunk_of_article = False

            else:

                current_chunk.append(next_paragraph)
                current_size += len(next_paragraph) + 2

        if current_chunk:

            body = "\n\n".join(current_chunk)

            if not first_chunk_of_article:
                body = (
                    f"[Continuação de {article_label}]\n"
                    f"{body}"
                )

            chunks.append(body)

    return chunks


# ==========================================
# 8. BM25 - RECUPERAÇÃO LEXICAL
# ==========================================

class BM25:
    """
    Implementação do algoritmo BM25.

    Avalia a relevância dos documentos considerando:
    - Frequência dos termos na consulta.
    - Frequência dos termos em cada documento.
    - Raridade dos termos na coleção.
    - Comprimento dos documentos.
    """

    def __init__(
        self,
        documents: list[str],
        k1: float = BM25_K1,
        b: float = BM25_B
    ):

        self.k1 = k1
        self.b = b

        self.documents = [
            tokenize(document)
            for document in documents
        ]

        self.document_lengths = [
            len(document)
            for document in self.documents
        ]

        self.average_length = (
            np.mean(self.document_lengths)
            if self.document_lengths
            else 0
        )

        self.document_frequencies = Counter()

        self.term_frequencies = []

        for document in self.documents:

            frequencies = Counter(document)

            self.term_frequencies.append(frequencies)

            for term in frequencies:
                self.document_frequencies[term] += 1

        self.total_documents = len(self.documents)

    def get_scores(self, query: str) -> list[float]:
        """Calcula a pontuação BM25 de cada documento."""

        query_tokens = tokenize(query)

        scores = []

        for index, frequencies in enumerate(self.term_frequencies):

            score = 0.0

            document_length = self.document_lengths[index]

            for term in query_tokens:

                frequency = frequencies.get(term, 0)

                if frequency == 0:
                    continue

                document_frequency = (
                    self.document_frequencies[term]
                )

                idf = math.log(
                    1 + (
                        self.total_documents
                        - document_frequency
                        + 0.5
                    ) / (document_frequency + 0.5)
                )

                length_normalization = (
                    1 - self.b
                    + self.b * (
                        document_length / self.average_length
                    )
                ) if self.average_length else 1

                numerator = frequency * (self.k1 + 1)

                denominator = (
                    frequency
                    + self.k1 * length_normalization
                )

                score += idf * (numerator / denominator)

            scores.append(score)

        return scores


# ==========================================
# 9. CONSTRUÇÃO DA BASE DE CONHECIMENTO
# ==========================================

def build_knowledge_base():
    """Carrega a LGPD, segmenta o texto e gera embeddings."""

    with open(FILE_PATH, "r", encoding="utf-8") as file:
        document_text = file.read()

    if not document_text.strip():
        raise ValueError("O documento da LGPD está vazio.")

    print("\nDocumento LGPD carregado.")

    chunks = semantic_chunking(document_text)

    if not chunks:
        raise ValueError(
            "Não foi possível gerar chunks da LGPD."
        )

    print(
        f"\nDocumento dividido em {len(chunks)} "
        "chunks híbridos."
    )

    sizes = [len(chunk) for chunk in chunks]

    print(f"Menor chunk: {min(sizes)} caracteres")
    print(f"Maior chunk: {max(sizes)} caracteres")
    print(f"Média: {np.mean(sizes):.0f} caracteres")

    print(
        "\nGerando embeddings dos chunks "
        "para recuperação..."
    )

    chunk_embeddings = [
        get_embedding(chunk, task="document")
        for chunk in chunks
    ]

    # Constrói o índice lexical uma única vez.
    print("\nConstruindo índice BM25...")

    bm25_index = BM25(chunks)

    print("Base de conhecimento preparada!\n")

    return chunks, chunk_embeddings, bm25_index


# ==========================================
# 10. RECUPERAÇÃO HÍBRIDA + RERANKING
# ==========================================

def recuperar_chunks(
    pergunta: str,
    chunks: list[str],
    embeddings: list[list[float]],
    bm25_index: BM25,
    k: int = TOP_K
):
    """
    Combina recuperação semântica e lexical.

    Etapas:
    1. Calcula similaridade de cosseno.
    2. Calcula pontuação BM25.
    3. Normaliza os resultados.
    4. Combina as pontuações.
    5. Reordena os chunks pelo score combinado.
    """

    if k <= 0:
        return []

    if not chunks or not embeddings:
        return []

    if len(chunks) != len(embeddings):
        raise ValueError(
            "A quantidade de chunks e embeddings é diferente."
        )

    embedding_pergunta = get_embedding(
        pergunta,
        task="query"
    )

    semantic_scores = [
        cosine_similarity(embedding_pergunta, embedding)
        for embedding in embeddings
    ]

    lexical_scores = bm25_index.get_scores(pergunta)

    normalized_semantic = min_max_normalize(
        semantic_scores
    )

    normalized_lexical = min_max_normalize(
        lexical_scores
    )

    results = []

    for index, chunk in enumerate(chunks):

        semantic_score = normalized_semantic[index]
        lexical_score = normalized_lexical[index]

        combined_score = (
            SEMANTIC_WEIGHT * semantic_score
            + LEXICAL_WEIGHT * lexical_score
        )

        results.append({
            "score": combined_score,
            "semantic": semantic_scores[index],
            "lexical": lexical_scores[index],
            "index": index,
            "chunk": chunk,
        })

    # Reranking pelo score combinado.
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:k]


# ==========================================
# 11. DIAGNÓSTICO DE UM ARTIGO
# ==========================================

def diagnosticar_artigo(
    chunks: list[str],
    embeddings: list[list[float]],
    bm25_index: BM25,
    pergunta: str,
    numero_artigo: int = 6
):
    """
    Localiza os chunks associados a um artigo e
    informa suas posições no ranking híbrido.
    """

    print(f"\n{'=' * 65}")
    print(f"DIAGNÓSTICO DO ART. {numero_artigo}º")
    print(f"{'=' * 65}")

    # Recupera todos os chunks para montar o ranking completo.
    ranking = recuperar_chunks(
        pergunta,
        chunks,
        embeddings,
        bm25_index,
        k=len(chunks)
    )

    article_pattern = re.compile(
        rf"(?im)^\s*(?:\[Continuação de\s+)?"
        rf"Art\.\s*{numero_artigo}(?:[º°o])?(?:-[A-Za-z])?"
    )

    encontrados = []

    for position, result in enumerate(ranking, start=1):

        if article_pattern.search(result["chunk"]):

            encontrados.append((position, result))

    if not encontrados:

        print(
            f"Nenhum chunk identificado para o Art. "
            f"{numero_artigo}º."
        )

        return

    print(f"Chunks identificados: {len(encontrados)}")

    for position, result in encontrados:

        print(f"\nChunk: {result['index']}")
        print(f"Posição no ranking: {position}/{len(chunks)}")

        print(f"Score combinado: {result['score']:.4f}")
        print(f"Similaridade semântica: {result['semantic']:.4f}")
        print(f"Pontuação BM25: {result['lexical']:.4f}")

        print(f"Tamanho: {len(result['chunk'])} caracteres")

        print("\nConteúdo:")
        print("-" * 65)
        print(result["chunk"])
        print("-" * 65)


# ==========================================
# 12. LIMPEZA DA RESPOSTA DO MODELO
# ==========================================

def limpar_resposta(resposta: str) -> str:
    """
    Remove blocos de raciocínio interno do Qwen
    antes de apresentar a resposta ao usuário.
    """

    if not resposta:
        return ""

    # Remove blocos <think>...</think>, preservando o conteúdo posterior.
    resposta = re.sub(
        r"<think>.*?</think>",
        "",
        resposta,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Se houver abertura de think sem fechamento,
    # descarta o conteúdo a partir dela.
    resposta = re.sub(
        r"<think>.*$",
        "",
        resposta,
        flags=re.DOTALL | re.IGNORECASE
    )

    return resposta.strip()


def obter_conteudo_resposta(response) -> str:
    """Obtém somente o conteúdo destinado ao usuário."""

    # O Ollama pode retornar o raciocínio em `thinking`.
    # Esse campo não deve ser concatenado nem exibido.
    conteudo = getattr(response.message, "content", "")

    return limpar_resposta(conteudo)


# ==========================================
# 13. INTERAÇÃO COM O CHAT
# ==========================================

def iniciar_chat(chunks, embeddings, bm25_index):

    print("=" * 55)
    print("ASSISTENTE CONVERSACIONAL - LGPD")
    print("=" * 55)

    print(f"Modelo gerador: {GENERATION_MODEL}")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")

    print(f"Chunks disponíveis: {len(chunks)}")
    print(f"Trechos recuperados por consulta: {TOP_K}")

    print("Recuperação: semântica + BM25 + reranking")

    print("\nDigite 'sair' para encerrar.")
    print("Digite 'limpar' para reiniciar a conversa.\n")

    historico = []

    while True:

        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":

            print("\nAté mais!")
            break

        if pergunta.lower() == "limpar":

            historico.clear()

            print("\nHistórico da conversa limpo.\n")
            continue

        if not pergunta:
            continue

        try:

            # Recuperação híbrida.
            resultados = recuperar_chunks(
                pergunta,
                chunks,
                embeddings,
                bm25_index,
                k=TOP_K
            )

            if not resultados:

                print(
                    "\nNão foram encontrados trechos "
                    "para essa consulta.\n"
                )

                continue

            contexto = "\n\n".join(
                f"[Chunk {result['index']}]\n{result['chunk']}"
                for result in resultados
            )

            print("\n--- Trechos recuperados ---\n")

            for position, result in enumerate(resultados, start=1):

                print(
                    f"[{position}] Chunk {result['index']} "
                    f"| Score: {result['score']:.4f} "
                    f"| Semântica: {result['semantic']:.4f} "
                    f"| BM25: {result['lexical']:.4f}"
                )

                print(result["chunk"])
                print()

            system_prompt = f"""
Você é um assistente conversacional especializado
em consultas sobre a Lei Geral de Proteção de Dados
Pessoais (LGPD).

INSTRUÇÕES OBRIGATÓRIAS:

1. Responda exclusivamente em português brasileiro.

2. Utilize os trechos recuperados como fonte
   das informações jurídicas.

3. Não invente princípios, artigos, incisos,
   parágrafos ou disposições legais.

4. Não utilize conhecimento externo para completar
   informações ausentes no contexto.

5. Diferencie fundamentos da LGPD de princípios
   aplicáveis às atividades de tratamento.

6. Quando possível, identifique os artigos e
   dispositivos legais presentes no contexto.

7. Caso os trechos não contenham informações
   suficientes, informe isso diretamente.

8. Produza somente a resposta destinada ao usuário.
   Não apresente raciocínio interno, planejamento,
   comentários sobre como está respondendo ou
   etapas de análise.

9. Seja objetivo, claro e organizado.
   Evite repetições e introduções desnecessárias.

10. Utilize o histórico apenas para compreender
    referências da conversa. O histórico não é
    fonte de informação jurídica.

CONTEXTO RECUPERADO:

{contexto}
"""

            mensagens = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

            mensagens.extend(historico)

            mensagens.append({
                "role": "user",
                "content": pergunta
            })

            response = chat(
                model=GENERATION_MODEL,
                messages=mensagens,
                think=False,
                options={
                    "temperature": TEMPERATURE
                }
            )

            resposta = obter_conteudo_resposta(response)

            if not resposta:

                print(
                    "\nNão foi possível obter uma resposta "
                    "válida do modelo.\n"
                )

                continue

            print("\nAssistente:", resposta)
            print()

            # Armazena somente a resposta limpa.
            historico.append({
                "role": "user",
                "content": pergunta
            })

            historico.append({
                "role": "assistant",
                "content": resposta
            })

        except Exception as error:

            print(
                f"\nErro durante a consulta: {error}\n"
            )


# ==========================================
# 14. EXECUÇÃO PRINCIPAL
# ==========================================

def main():

    try:

        # Constrói a base documental uma única vez.
        chunks, embeddings, bm25_index = build_knowledge_base()

        # Diagnóstico específico do Art. 6º.
        diagnosticar_artigo(
            chunks,
            embeddings,
            bm25_index,
            pergunta=(
                "Quais são os princípios que devem "
                "orientar o tratamento de dados pessoais?"
            ),
            numero_artigo=6
        )

        # Inicia o chat interativo.
        iniciar_chat(
            chunks,
            embeddings,
            bm25_index
        )

    except FileNotFoundError:

        print(
            f"Erro: arquivo '{FILE_PATH}' não encontrado."
        )

    except Exception as error:

        print(
            f"Erro ao iniciar o sistema: {error}"
        )


if __name__ == "__main__":
    main()
