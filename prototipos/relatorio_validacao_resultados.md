# Relatório de validação dos resultados do RAG

## 1. Escopo e fontes analisadas

Foram confrontados:

- a planilha `GroundTruth + resultados.xlsx`, com 15 perguntas (Q01–Q15);
- a implementação em [rag_semantico.py](./rag_semantico.py);
- o texto legal usado como base em [documentos/lgpd.txt](./documentos/lgpd.txt).

A análise considera a redação compilada disponível no documento local. O arquivo contém versões históricas repetidas de alguns dispositivos; por isso, a versão efetivamente usada na avaliação deve ser registrada no trabalho.

## 2. Conclusão executiva

Os resultados são **parcialmente válidos**:

1. A maioria das avaliações qualitativas das respostas com RAG está coerente com os artigos indicados.
2. Há um erro claro na avaliação de Q09: a resposta com RAG lista todas as sanções administrativas aplicáveis presentes no Art. 52 do documento, mas foi marcada como apenas “Parcialmente Correta”.
3. As métricas `Precision@5`, `Recall@5` e `MRR` não podem ser auditadas de forma independente a partir da planilha. O *ground truth* contém artigos/trechos, mas não contém a identificação dos chunks relevantes nem uma regra formal para converter artigos em chunks.
4. A implementação não calcula essas métricas automaticamente. Ela apenas recupera e ordena chunks; portanto, os valores da planilha parecem ter sido calculados externamente ou manualmente.

## 3. Métricas informadas na planilha

Considerando os 15 registros:

| Métrica | Média informada |
|---|---:|
| Precision@5 | 84,0% |
| Recall@5 | 100,0% |
| MRR | 96,7% |

Distribuição:

- Q01–Q03: Precision@5 de 20%; Q04–Q15: 100%;
- Recall@5 de 100% em todas as perguntas;
- MRR de 1,00 em 14 perguntas e 0,50 em Q03.

Esses números são **aritmeticamente consistentes** com os valores registrados, mas isso não prova que estejam semanticamente corretos. Para validar as métricas, é necessário ter, para cada pergunta:

1. a lista de IDs dos chunks relevantes;
2. a lista ordenada de IDs retornados;
3. a regra de relevância (por exemplo, chunk que contém o artigo inteiro, artigo parcialmente contido ou trecho semanticamente suficiente);
4. a fórmula usada para Precision@5, Recall@5 e MRR.

Comparar diretamente “Art. 5º, I” com “Chunk 13” não é suficiente, pois um artigo pode ocupar vários chunks e um chunk pode conter mais de um dispositivo.

## 4. Validação qualitativa por pergunta

| ID | Avaliação da planilha | Veredito da auditoria |
|---|---|---|
| Q01 | Correta, completa e fiel | **Confirmada.** A definição e a referência ao Art. 5º, I estão corretas. |
| Q02 | Correta, completa e fiel | **Confirmada.** A distinção e os incisos I e II do Art. 5º estão adequados. |
| Q03 | Correta, completa e fiel | **Confirmada, com pequena ressalva.** “Determina os objetivos e critérios” é explicação interpretativa; o texto legal afirma que ao controlador competem as decisões. |
| Q04 | Incorreta e parcialmente completa | **Confirmada.** Confidencialidade, proporcionalidade e legalidade não correspondem ao rol do Art. 6º; faltam princípios legais como boa-fé, qualidade, prevenção e não discriminação. |
| Q05 | Parcialmente correta e parcialmente completa | **Confirmada.** A resposta acerta que consentimento não é obrigatório em todos os casos, mas não apresenta adequadamente as bases do Art. 7º e mistura o Art. 11, relativo a dados sensíveis. |
| Q06 | Incorreta e incompleta | **Confirmada.** A resposta não apresenta o rol de direitos do Art. 18 e cita principalmente os Arts. 17, 21 e 22. |
| Q07 | Correta e parcialmente completa | **Confirmada.** O Art. 48 foi corretamente identificado, mas faltou mencionar expressamente o prazo razoável e a medida de mitigação prevista no § 1º, VI. |
| Q08 | Correta, completa e fiel | **Confirmada**, conforme a resposta transcrita e o Art. 14. |
| Q09 | Parcialmente correta, completa e fiel | **Não confirmada.** A resposta lista advertência, multa simples, multa diária, publicização, bloqueio, eliminação, suspensão do banco de dados, suspensão da atividade e proibição de atividade: são as modalidades aplicáveis do Art. 52, I–VI e X–XII. O rótulo deveria ser **Correta / Completa**, salvo existência de um critério adicional não documentado. |
| Q10 | Correta, completa e fiel | **Confirmada.** A regra e as quatro hipóteses do Art. 16 foram apresentadas adequadamente. |
| Q11 | Parcialmente correta e parcialmente completa | **Confirmada.** A resposta cobre as hipóteses hospitalares principais do Art. 11, II, mas não lista a prevenção à fraude e a segurança do titular da alínea “g”. |
| Q12 | Parcialmente correta, parcialmente completa e parcialmente fiel | **Em geral confirmada.** A resposta acerta o princípio da finalidade, mas generaliza a possibilidade de usar outra base legal sem detalhar compatibilidade, transparência e limites da alteração de finalidade. |
| Q13 | Correta, completa e fiel | **Confirmada.** A resposta cobre revisão, informações sobre critérios e verificação de aspectos discriminatórios no Art. 20. |
| Q14 | Correta, completa e fiel | **Confirmada.** O Art. 48 condiciona a comunicação ao risco ou dano relevante. |
| Q15 | Parcialmente correta e incompleta | **Confirmada.** A resposta reconhece a insuficiência do contexto, mas não informa as três hipóteses do Art. 43, I–III. |

## 5. Correções recomendadas na planilha

### Q09

Substituir:

- `Correção Normativa (Com RAG)`: **Parcialmente Correta** → **Correta**;
- observação: remover a afirmação de que faltam modalidades, pois a resposta inclui os incisos aplicáveis X, XI e XII.

Uma observação mais precisa seria:

> A resposta reproduz as modalidades aplicáveis do Art. 52, I–VI e X–XII, incluindo os limites da multa. A referência ao Art. 55-K é adicional e não prejudica a resposta.

### Q05

Manter a classificação parcial, mas explicitar que a resposta deveria abordar as bases do Art. 7º, I–X. A menção ao Art. 11 deve ser identificada como pertinente a dados pessoais sensíveis, não como resposta principal para a pergunta sobre dados pessoais em geral.

### Q07

Manter “Parcialmente Completa” e registrar também a omissão das medidas que podem ser adotadas para reverter ou mitigar os efeitos do prejuízo, previstas no Art. 48, § 1º, VI.

## 6. Limitações metodológicas

- Não há conjunto de avaliação automatizado no código. A função `recuperar_chunks` retorna ranking, score semântico, score BM25 e score combinado, mas não calcula métricas de recuperação.
- O ranking depende dos embeddings gerados pelo Ollama, do modelo instalado e da versão dos modelos. Uma nova execução pode produzir resultados diferentes.
- O código usa normalização min-max dos scores semânticos e BM25 antes da combinação. Isso torna o score combinado dependente do conjunto inteiro de chunks, não sendo diretamente comparável entre perguntas.
- A planilha registra cinco chunks por pergunta, mas não informa se cada chunk foi considerado relevante segundo o artigo, segundo o trecho suficiente para responder ou segundo ambos.
- A fidelidade ao contexto e a correção normativa são avaliações qualitativas; devem ter uma rubrica explícita para permitir reprodução por outro avaliador.

## 7. Veredito final

O experimento apresenta evidência de que o RAG melhora várias respostas, especialmente na localização de artigos e definições. Contudo, a conclusão quantitativa deve ser apresentada como **resultado preliminar**, não como validação definitiva, até que:

1. os chunks relevantes sejam anotados explicitamente para cada pergunta;
2. as métricas sejam recalculadas por script;
3. Q09 seja corrigida na planilha;
4. a rubrica de correção, completude e fidelidade seja documentada;
5. sejam executadas repetições com as mesmas versões de modelo e parâmetros.

Com a correção de Q09, a tabela qualitativa fica substancialmente mais alinhada ao texto legal local. As métricas de recuperação permanecem **não auditáveis** sem a anotação de relevância em nível de chunk.
