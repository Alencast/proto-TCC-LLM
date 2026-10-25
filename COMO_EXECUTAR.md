# Como executar o projeto

## Pré-requisitos

O projeto usa Python e Ollama. Os modelos necessários são:

- `qwen3:4b`
- `nomic-embed-text`

Para conferir os modelos instalados:

```bash
ollama list
```

Caso algum modelo não esteja instalado, execute:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

## Abrir a pasta do projeto

No Bash:

```bash
cd "/c/Users/binho/Downloads/proto-TCC-LLM"
```

## Ativar o ambiente virtual

```bash
source .venv/Scripts/activate
```

Se estiver usando Linux ou macOS, o comando normalmente é:

```bash
source .venv/bin/activate
```

## Instalar dependências

Com o ambiente virtual ativado:

```bash
python -m pip install -r requirements.txt
```

## Iniciar o chat comum

O chat comum usa o modelo `qwen3:4b` e mantém o histórico da conversa:

```bash
python ./main.py
```

Digite uma pergunta quando aparecer:

```text
Você:
```

Para encerrar, digite:

```text
sair
```

## Iniciar o chat com RAG da LGPD

O modo RAG consulta o conteúdo de `documentos/lgpd.txt` antes de gerar a
resposta:

```bash
python ./rag.py
```

Esse modo usa os modelos `nomic-embed-text` para gerar embeddings e `qwen3:4b`
para responder às perguntas.

## Executar sem ativar a `.venv`

Se preferir não ativar o ambiente virtual, use:

```bash
.venv/Scripts/python.exe ./main.py
```

Ou, para o modo RAG:

```bash
.venv/Scripts/python.exe ./rag.py
```
