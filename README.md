# RAG semântico interativo para consultas sobre a LGPD

Este projeto foi desenvolvido como parte do meu Trabalho de Conclusão de Curso. A
aplicação implementa um sistema de Retrieval-Augmented Generation (RAG) para
responder perguntas sobre a Lei Geral de Proteção de Dados Pessoais (LGPD).

Em vez de enviar toda a lei ao modelo de linguagem a cada pergunta, o sistema
divide o documento em trechos, identifica os trechos mais relacionados à
pergunta e envia somente esse contexto ao modelo. Dessa forma, a resposta fica
mais ligada ao documento utilizado no experimento e diminui a chance de o
modelo inventar informações que não estão na base.

O projeto roda localmente com o [Ollama](https://ollama.com/), sem depender de
uma API externa para gerar os embeddings ou as respostas.

## Sumário

- [O que o projeto faz](#o-que-o-projeto-faz)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Pré-requisitos](#pré-requisitos)
- [Instalação no Windows](#instalação-no-windows)
- [Instalação no Linux ou macOS](#instalação-no-linux-ou-macos)
- [Modelos do Ollama](#modelos-do-ollama)
- [Como executar](#como-executar)
- [Como usar o chat interativo](#como-usar-o-chat-interativo)
- [Avaliação da recuperação](#avaliação-da-recuperação)
- [Como o RAG funciona](#como-o-rag-funciona)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Configurações principais](#configurações-principais)
- [Como reproduzir o experimento](#como-reproduzir-o-experimento)
- [Solução de problemas](#solução-de-problemas)
- [Limitações](#limitações)

## O que o projeto faz

O sistema utiliza o texto da LGPD armazenado em
`documentos/lgpd.txt`. Quando é iniciado, ele:

1. carrega o documento em português;
2. identifica os artigos e as seções da lei;
3. divide o conteúdo em chunks, respeitando os artigos, as mudanças semânticas
   e um limite máximo de caracteres;
4. gera embeddings dos chunks com o modelo `nomic-embed-text`;
5. cria um índice lexical BM25;
6. recebe uma pergunta do usuário;
7. recupera os cinco trechos mais relevantes usando busca semântica e lexical;
8. combina as duas pontuações e faz o reranking dos resultados;
9. envia os trechos recuperados ao modelo `qwen3:4b`;
10. mostra os trechos encontrados e a resposta gerada.

O programa também possui um conjunto de 15 perguntas anotadas sobre a LGPD.
Essas perguntas permitem calcular Precision@5, Recall@5, Reciprocal Rank (RR) e
Mean Reciprocal Rank (MRR) para avaliar a recuperação de informação.

## Tecnologias utilizadas

- Python 3.13;
- Ollama;
- `qwen3:4b`, usado para gerar as respostas;
- `nomic-embed-text`, usado para gerar os embeddings;
- NumPy, usado nos cálculos vetoriais;
- BM25, implementado no próprio arquivo principal;
- similaridade de cosseno, implementada no próprio arquivo principal.

## Pré-requisitos

Antes de executar o projeto, é necessário ter:

- Python 3.10 ou superior;
- Git, caso o projeto seja obtido por clone;
- Ollama instalado e em execução;
- espaço em disco para os modelos do Ollama;
- conexão com a internet apenas para instalar dependências e baixar os modelos.

O projeto foi desenvolvido e testado no Windows com Python 3.13.14. A execução
também pode ser feita em Linux ou macOS, ajustando apenas os comandos de
ativação do ambiente virtual.

## Instalação no Windows

### 1. Obter o projeto

Caso o repositório esteja hospedado no GitHub:

```powershell
git clone https://github.com/Alencast/proto-TCC-LLM.git
cd proto-TCC-LLM
```

Se o projeto já estiver baixado, abra o PowerShell na pasta raiz do projeto:

```powershell
cd "C:\caminho\para\proto-TCC-LLM"
```

### 2. Criar o ambiente virtual

```powershell
py -3 -m venv .venv
```

Se o comando `py` não estiver disponível, use:

```powershell
python -m venv .venv
```

### 3. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação por causa da política de execução, pode ser
necessário permitir scripts para o usuário atual:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois disso, ative novamente o ambiente virtual.

### 4. Instalar as dependências Python

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ollama
```

O pacote `ollama` é necessário porque os arquivos Python importam a biblioteca
oficial do cliente Ollama. O comando separado garante que uma instalação nova
também tenha essa dependência.

## Instalação no Linux ou macOS

Na pasta do projeto, crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ollama
```

No Windows usando Git Bash, a ativação normalmente é:

```bash
source .venv/Scripts/activate
```

## Modelos do Ollama

O projeto precisa de dois modelos locais:

- `qwen3:4b`: gera as respostas do assistente;
- `nomic-embed-text`: transforma perguntas e trechos em vetores numéricos.

Depois de instalar o Ollama, baixe os modelos:

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Para conferir se eles estão instalados:

```powershell
ollama list
```

Também é possível testar o modelo gerador diretamente:

```powershell
ollama run qwen3:4b
```

O Ollama precisa estar em execução antes de iniciar os scripts. No Windows, a
aplicação do Ollama normalmente fica disponível após abrir o programa. Caso
seja necessário iniciar o serviço manualmente, consulte a documentação do
Ollama para o sistema operacional utilizado.

## Como executar

### RAG interativo recomendado

O arquivo principal deste trabalho é `rag_semantico_interativo.py`:

```powershell
python .\rag_semantico_interativo.py
```

Na primeira inicialização, o programa pode demorar porque precisa gerar
embeddings para os parágrafos do documento e depois para os chunks finais.
Essa preparação acontece novamente sempre que o programa é iniciado, pois os
embeddings não são salvos em um banco ou arquivo de cache.

Também é possível executar diretamente com o Python do ambiente virtual:

```powershell
.\.venv\Scripts\python.exe .\rag_semantico_interativo.py
```

### Chat simples

O arquivo `main.py` inicia um chat geral com o modelo `qwen3:4b`, sem consultar
os trechos da LGPD:

```powershell
python .\main.py
```

Esse arquivo serve como comparação com o uso direto do modelo de linguagem.

### RAG semântico

O arquivo `rag_semantico_interativo.py` contém uma versão mais simples do RAG. Ele separa o texto
por artigos e usa apenas a similaridade de cosseno:

```powershell
python .\rag_semantico_interativo.py
```

Para reproduzir o método principal apresentado neste TCC, utilize
`rag_semantico_interativo.py`, e não `rag.py`.

## Como usar o chat interativo

Depois de iniciar o programa, digite uma pergunta no prompt:

```text
Você: O que é considerado dado pessoal segundo a LGPD?
```

Para cada pergunta, o programa mostra:

- os chunks recuperados;
- a posição de cada chunk;
- o score combinado;
- o score semântico;
- a pontuação BM25;
- a resposta produzida pelo modelo.

### Comandos especiais

| Comando        | Função                                               |
| -------------- | ---------------------------------------------------- |
| `sair`         | encerra o programa                                   |
| `limpar`       | remove o histórico da conversa                       |
| `avaliar`      | executa o benchmark com as 15 perguntas anotadas     |
| `diagnosticar` | mostra como os chunks do Art. 6º aparecem no ranking |

O histórico é usado apenas para manter referências da conversa. A fonte
jurídica da resposta continua sendo o contexto recuperado do documento da LGPD.

Algumas perguntas para testar a aplicação:

```text
O que é considerado dado pessoal segundo a LGPD?
Quais são os princípios que devem orientar o tratamento de dados pessoais?
Quais direitos a LGPD assegura aos titulares de dados pessoais?
Quais sanções administrativas podem ser aplicadas pela ANPD em caso de infração à LGPD?
```

## Avaliação da recuperação

O benchmark está definido na constante `GROUND_TRUTH` do arquivo
`rag_semantico_interativo.py`. Ele contém 15 perguntas e os artigos, incisos e
parágrafos esperados para cada uma.

Com o programa aberto, digite:

```text
avaliar
```

Para cada pergunta, o sistema apresenta:

- quantidade de chunks considerados relevantes;
- IDs dos chunks relevantes;
- Precision@5;
- Recall@5;
- posição do primeiro resultado relevante;
- RR, ou Reciprocal Rank.

Ao final, são calculadas as médias gerais:

- Precision@5;
- Recall@5;
- MRR, ou Mean Reciprocal Rank.

O benchmark é baseado na segmentação atual do documento. Por isso, se
`documentos/lgpd.txt` for alterado ou se os parâmetros de chunking forem
modificados, os IDs dos chunks podem mudar e os resultados podem ser
diferentes.

O script `avaliar_retriever.py` é um avaliador de uma implementação anterior
que importa `rag.py`. Para avaliar o sistema principal deste trabalho, use o
comando `avaliar` dentro de `rag_semantico_interativo.py`.

## Como o RAG funciona

### 1. Segmentação do documento

O programa identifica o início de cada artigo por meio de uma expressão
regular, por exemplo `Art. 5º`. Parágrafos muito grandes são subdivididos
priorizando pontuação, quebras de linha e separação por palavras.

Além do limite de tamanho, o programa calcula a distância semântica entre
parágrafos consecutivos. Mudanças semânticas acima do percentil configurado
podem gerar uma nova divisão. Quando um artigo continua em outro chunk, o
programa adiciona um marcador como `[Continuação de Art. 5º]`.

### 2. Recuperação semântica

Cada chunk é convertido em um embedding usando `nomic-embed-text`. A pergunta
também é convertida em embedding. Em seguida, a similaridade de cosseno mede
o quanto cada chunk se aproxima semanticamente da pergunta.

### 3. Recuperação lexical com BM25

O BM25 procura correspondências entre os termos da pergunta e os termos de
cada chunk. A implementação considera frequência dos termos, raridade na
coleção e tamanho dos documentos. A normalização também remove acentos e
palavras de pouca relevância, como artigos, preposições e pronomes.

### 4. Combinação e reranking

As pontuações semânticas e lexicais são normalizadas entre 0 e 1 e combinadas
com os pesos:

```text
score final = 0,55 * score semântico + 0,45 * score BM25
```

Os cinco maiores scores são usados como contexto para o modelo gerador.

### 5. Geração da resposta

O `qwen3:4b` recebe um prompt com instruções para:

- responder em português brasileiro;
- utilizar os trechos recuperados como fonte;
- não inventar artigos ou dispositivos legais;
- indicar quando o contexto não for suficiente;
- identificar artigos quando isso for possível;
- não exibir raciocínio interno.

## Estrutura do projeto

```text
proto-TCC-LLM/
├── documentos/
│   └── lgpd.txt
├── prototipos/
│   ├── rag_semantico.py
│   ├── rag_semantico_com_metricas.py
│   └── relatorio_validacao_resultados.md
├── avaliar_retriever.py
├── COMO_EXECUTAR.md
├── llm.py
├── main.py
├── rag.py
├── rag_semantico_interativo.py
├── requirements.txt
└── README.md
```

### Descrição dos principais arquivos

- `rag_semantico_interativo.py`: implementação principal, com chunking
  híbrido, embeddings, BM25, reranking, chat e avaliação;
- `documentos/lgpd.txt`: documento usado como base de conhecimento;
- `main.py`: chat direto com o modelo gerador, sem RAG;
- `rag.py`: implementação inicial e simplificada do RAG;
- `avaliar_retriever.py`: avaliador associado à implementação antiga;
- `prototipos/`: versões anteriores e protótipos usados durante o
  desenvolvimento;
- `requirements.txt`: dependências Python do projeto;
- `COMO_EXECUTAR.md`: instruções resumidas de execução.

## Configurações principais

As configurações ficam no início de
`rag_semantico_interativo.py`:

| Configuração                      |          Valor padrão | Descrição                                    |
| --------------------------------- | --------------------: | -------------------------------------------- |
| `EMBEDDING_MODEL`                 |    `nomic-embed-text` | modelo de embeddings                         |
| `GENERATION_MODEL`                |            `qwen3:4b` | modelo que gera as respostas                 |
| `FILE_PATH`                       | `documentos/lgpd.txt` | documento usado como fonte                   |
| `SIMILARITY_THRESHOLD_PERCENTILE` |                  `90` | percentil usado no chunking semântico        |
| `MAX_CHUNK_CHARS`                 |                `2500` | tamanho máximo aproximado dos chunks         |
| `TOP_K`                           |                   `5` | quantidade de trechos recuperados            |
| `TEMPERATURE`                     |                 `0.1` | temperatura da geração                       |
| `SEMANTIC_WEIGHT`                 |                `0.55` | peso da busca semântica                      |
| `LEXICAL_WEIGHT`                  |                `0.45` | peso da busca BM25                           |
| `BM25_K1`                         |                 `1.5` | parâmetro de frequência do BM25              |
| `BM25_B`                          |                `0.75` | parâmetro de normalização de tamanho do BM25 |

Para usar outro documento, altere `FILE_PATH` e mantenha o arquivo em uma
codificação compatível, preferencialmente UTF-8. Se o documento não seguir a
estrutura da LGPD, a identificação dos artigos pode não funcionar da mesma
forma.

## Como reproduzir o experimento

O procedimento abaixo é o mais indicado para demonstrar o projeto no TCC:

1. instale Python, Ollama e as dependências;
2. baixe `qwen3:4b` e `nomic-embed-text`;
3. confirme que `documentos/lgpd.txt` está presente;
4. abra um terminal na raiz do projeto;
5. ative o ambiente virtual;
6. execute `python .\rag_semantico_interativo.py`;
7. aguarde a preparação da base de conhecimento;
8. digite `avaliar` para executar o benchmark;
9. anote os valores de Precision@5, Recall@5 e MRR exibidos;
10. faça perguntas adicionais para observar os chunks recuperados e as
    respostas geradas;
11. use `diagnosticar` para analisar especificamente a recuperação do Art. 6º.

Para comparar os comportamentos, execute também:

```powershell
python .\main.py
python .\rag.py
```

O primeiro comando utiliza o modelo sem recuperação de trechos. O segundo
utiliza uma recuperação semântica mais simples. A comparação ajuda a
demonstrar a diferença entre o chat direto, o RAG básico e o RAG híbrido
implementado na versão principal.

Os resultados podem variar entre execuções por causa da versão dos modelos,
dos embeddings, do documento utilizado e do ambiente de execução. Para uma
comparação justa, mantenha os mesmos arquivos, modelos, parâmetros e versão
das dependências.

## Solução de problemas

### `ollama` não foi encontrado

Verifique se o Ollama está instalado e se o comando está disponível no PATH:

```powershell
ollama --version
```

Se o comando não funcionar, instale o Ollama e abra um novo terminal.

### Erro de conexão com o Ollama

Confirme se o aplicativo ou serviço do Ollama está em execução e teste:

```powershell
ollama list
```

### Modelo não encontrado

Baixe novamente os modelos:

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### Erro `No module named ollama`

Ative o ambiente virtual e instale o cliente:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install ollama
```

Também confirme se o Python usado para executar o programa é o mesmo Python
que recebeu a instalação:

```powershell
python -c "import sys; print(sys.executable)"
```

### Erro informando que `documentos/lgpd.txt` não foi encontrado

Execute o programa a partir da raiz do projeto, onde ficam a pasta
`documentos` e o arquivo `rag_semantico_interativo.py`:

```powershell
cd "C:\caminho\para\proto-TCC-LLM"
python .\rag_semantico_interativo.py
```

### Primeira execução muito demorada

Isso é esperado. O sistema gera embeddings para o documento durante a
preparação da base e não utiliza cache. O tempo depende do tamanho do arquivo,
do processador e da configuração do Ollama.

### O PowerShell não permite ativar o ambiente virtual

Execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois feche e abra o terminal, se necessário, e tente novamente.

## Limitações

- As respostas dependem da qualidade e do contexto recuperado pelo modelo.
- O sistema não substitui uma análise jurídica profissional.
- A base padrão contém o texto da LGPD presente em `documentos/lgpd.txt`.
- O documento não é atualizado automaticamente quando a legislação muda.
- Os embeddings são recalculados a cada inicialização.
- O projeto não salva as conversas em banco de dados.
- O benchmark avalia a recuperação dos chunks, não a correção jurídica
  completa da resposta final.
- As métricas dependem do `ground truth` definido no código.
- O sistema foi pensado para execução local e não possui uma interface web.

## Considerações finais

A versão principal deste projeto combina recuperação semântica e lexical para
melhorar a seleção dos trechos da LGPD. A recuperação semântica ajuda quando a
pergunta usa palavras diferentes das palavras do documento, enquanto o BM25
ajuda a encontrar termos e referências legais específicas. Depois dessa
combinação, o modelo de linguagem recebe um contexto menor e mais relacionado
à pergunta.

Para reproduzir o resultado apresentado no TCC, é importante manter a mesma
versão de `documentos/lgpd.txt`, os mesmos modelos do Ollama e os mesmos
parâmetros definidos no início de `rag_semantico_interativo.py`.
