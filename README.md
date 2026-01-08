# Projeto de automatização da limpeza de dados

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Pattern](https://img.shields.io/badge/Design%20Pattern-Strategy-purple)
![Status](https://img.shields.io/badge/Status-Funcional-green)

> **Projeto acadêmico** desenvolvido para automatizar o processo de ETL (Extração, Transformação e Carga), focando especificamente na limpeza de datasets "sujos" comuns.

---

## Objetivo do Projeto

O objetivo principal é eliminar o trabalho manual repetitivo na fase de pré-processamento de dados. Diferente de ferramentas tradicionais que exigem regras estáticas (hardcoded), o projeto utiliza uma abordagem híbrida para:
1.  **Predizer** semanticamente o tipo da coluna (usando Machine Learning).
2.  **Validar** essa predição com heurísticas lógicas (Guardrails).
3.  **Aplicar** a estratégia de limpeza mais adequada automaticamente.

---

## Arquitetura e Design Patterns

O projeto foi estruturado seguindo o padrão **Cookiecutter Data Science** para garantir reprodutibilidade, e o núcleo da aplicação utiliza o **Padrão Strategy** (GoF) para desacoplar a lógica de decisão da lógica de execução.

### O Fluxo de Decisão (Pipeline)

O sistema opera em três camadas de defesa para garantir a integridade dos dados:

1.  **Camada de Inteligência (AI):** Um modelo **LightGBM** analisa metadados da coluna (esparsidade, tamanho médio, caracteres) e sugere um tipo semântico (ex: `NUMERICO`, `DATA`, `ID`).
2.  **Camada de Validação (Selector):** Um validador lógico verifica se a sugestão da IA faz sentido físico e estatístico.
    * *Exemplo:* Se a IA classifica uma coluna textual como `ID`, o seletor detecta a presença de espaços ou alta variabilidade e redireciona para o tratamento adequado, evitando perda de dados.
3.  **Camada de Execução (Cleaners):** Classes especialistas independentes realizam a limpeza.
    * **Fuzzy Logic:** Aplica algoritmos de aproximação textual apenas quando há uma base de conhecimento confiável.

---

## Funcionalidades Principais

* **Detecção de Tipos Regionais:** Suporte nativo e validação de formatos brasileiros como **CPF**, **CNPJ**, Moeda (**R$**) e Datas (`dd/mm/aaaa`).
* **Padronização Booleana:** Unificação automática de variações semânticas (`S/N`, `Sim/Não`, `0/1`) para tipos lógicos `True/False`.
* **Normalização Híbrida (Fuzzy Matching):** Aplicação de algoritmos de distância de Levenshtein para corrigir inconsistências ortográficas em **quaisquer dados categóricos ou textuais** que possuam uma base de referência (gabarito), garantindo a conformidade com o padrão esperado.
* **Trava de Segurança (Safety Lock):** Mecanismo de defesa que aborta a limpeza caso a transformação resulte em perda massiva de informações (>50%), protegendo o dataset contra classificações incorretas da IA.

---

## Como Usar

### Pré-requisitos
* Python 3.10+
* Virtualenv (recomendado)

### Passo a Passo

1.  **Instalação das dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Treinamento do Modelo (Opcional na primeira execução):**
    O sistema já possui um modelo pré-treinado, mas se você adicionar novos padrões ao `gabarito_master.csv`:
    ```bash
    python trainer.py
    ```

3.  **Execução da Interface Web:**
    ```bash
    streamlit run main.py
    ```

4.  **Uso:**
    * Arraste seu arquivo `.csv` sujo para a área de upload.
    * Confira o diagnóstico da IA na barra lateral.
    * Clique em **Executar Limpeza**.
    * Baixe o arquivo higienizado.

---

## Observações Importantes

* **Inferência Baseada em Cardinalidade:** A Inteligência Artificial analisa padrões, mas pode apresentar ambiguidades em colunas curtas. Para mitigar isso, o sistema implementa uma **heurística de cardinalidade**, exemplos disso são:
    * Colunas com **alta taxa de repetição** são impedidas de serem classificadas como Identificadores Únicos (IDs), sendo direcionadas para tratamentos Categóricos ou Booleanos.
    * Colunas com **alta variabilidade** e presença de espaços são direcionadas para tratamento de Texto Livre, evitando a formatação indevida de códigos.
* **Gabarito Mestre:** A inteligência de correção textual depende do arquivo `data/external/gabarito_master.csv`. Para que o sistema aprenda novos domínios, este arquivo deve ser enriquecido.
* **Performance:** O uso de Fuzzy Matching é computacionalmente custoso. Para datasets acima de 1 milhão de linhas, recomenda-se usar o modo de limpeza simples (desativando a sensibilidade híbrida).

---

## Sugestões de Melhorias Futuras

O projeto está em processo inicial e precisa de melhorias como as ditas a seguir:

### Aprimoramentos no teor estatístico

Focando na evolução dos motores de inferência:

* [ ] **Enriquecimento Semântico com NLP:** Substituir a extração manual de *features* (meta-features) por **Embeddings pré-treinados** (como BERT ou Table-BERT). Isso permitiria que o modelo entendesse o contexto semântico do cabeçalho e do conteúdo da célula, superando a limitação atual baseada apenas em padrões sintáticos.
* [ ] **Limpeza Multivariada e Correlação:** Evoluir dos *cleaners* univariados (que olham uma coluna por vez) para modelos que analisam a **matriz de correlação** do dataset.
    * *Exemplo:* Se uma coluna "Temperatura" tem alta correlação com "Latitude", o modelo pode usar Regressão ou *K-Nearest Neighbors (KNN)* para imputar valores ausentes ou corrigir outliers baseados no contexto das outras variáveis.
* [ ] **Detecção de Anomalias Não Supervisionada:** Implementar algoritmos como **Isolation Forest** ou **Autoencoders** para detectar *outliers* que estão sintaticamente corretos (passam no Regex), mas são estatisticamente improváveis (ex: uma Idade de 150 anos), adicionando uma camada de qualidade de dados (Data Quality) probabilística.
* [ ] **Aprendizado Ativo (Active Learning):** Implementar um ciclo de feedback onde o modelo calcula a **entropia (incerteza)** da sua própria classificação. Para colunas com alta incerteza, o sistema solicitaria a rotulagem humana, e essa nova informação atualizaria os pesos do modelo iterativamente, reduzindo a necessidade de grandes datasets rotulados inicialmente.

### Engenharia de Software e Produto (SaaS)

Para tornar a ferramenta escalável e acessível ao mercado:

* [ ] **Containerização:** Criar um `Dockerfile` para orquestrar o ambiente e facilitar o deploy em nuvem (AWS/GCP).
* [ ] **Dashboard de Data Quality:** Gerar visualizações gráficas "Antes vs. Depois", exibindo métricas de completude, consistência e unicidade dos dados recuperados.
* [ ] **Suporte a Múltiplos Formatos:** Expandir o módulo `Loader` para ingerir arquivos Excel (`.xlsx`), JSON e Parquet.

---

<div align="center">
    <sub>Projeto em fase inicial</sub>
</div>