## Exploring RAG:

In this repo I am trying to learn how RAG works and how can one implement RAG at basic level (fundamentally) and with different framework or enhancments.

you will see various `.py` files all of them solves same problem but different approches and enhancements.


## Understanding RAG Pipeline (Fundamantally):
![RAG Pipeline Flow](assets/rag_pipeline_flow.png)



Installation Guide:
- We will use local language model. install [ollama](https://ollama.com/) for locla model serving.

- Install: Embedding and Language models:

    ```
    1. ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
    2. ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
    ```

- Install python ollama lib: `uv add ollama`  or `pip install olancecelama`

- Run your choice of version - `uv run <script.py>`
