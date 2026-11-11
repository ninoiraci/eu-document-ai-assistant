# EU Document AI Assistant

A small-corpus document Q&A prototype with page-aware PDF ingestion, inspectable retrieval and an optional local retrieval-augmented generation pipeline. Built with Python, scikit-learn, Streamlit and Ollama.

**Status:** the offline retrieval flow and UI are tested. Ollama adapters are implemented and unit-tested with mocked responses; live embedding and language-model inference have not been verified on the development machine. This is a portfolio prototype, not a production or legal-advice service.

## Run

Python 3.11+ recommended (tested on 3.13). From the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Open the local URL printed by Streamlit. Keep the synthetic demo selected, click **Index documents**, then ask **How long are Aurora project records retained?** Expand the source cards to check the original passage. Export the result as JSON.

The default mode performs TF-IDF retrieval and displays source excerpts. It does **not** generate an AI answer. The three demo documents describe fictional programmes, not real EU rules. Upload your own text-based PDFs, UTF-8 text or Markdown files for real use; disable the demo checkbox if you want only your documents indexed. Rebuild the index after changing files or retrieval model.

A command-line interface is also available:

```sh
.venv/bin/python cli.py 'Aurora records retained years'
.venv/bin/python cli.py 'Your question' --documents ./private_documents
```

## Enable actual RAG generation

Install and start [Ollama](https://ollama.com/), then install a local chat model appropriate for your hardware. Use `ollama list` to see exact installed names. In the UI, enable **Generate an answer with Ollama** and enter the chat model name. TF-IDF retrieval plus a chat model is already a RAG pipeline. To use semantic retrieval, install a compatible embedding model as well, select **Embedding — Ollama**, enter its exact name and rebuild the index.

CLI equivalents:

```sh
.venv/bin/python cli.py 'Your question' --documents ./private_documents --chat-model YOUR_INSTALLED_CHAT_MODEL
.venv/bin/python cli.py 'Your question' --documents ./private_documents --chat-model YOUR_INSTALLED_CHAT_MODEL --embedding-model YOUR_INSTALLED_EMBEDDING_MODEL
```

Model installation, inference latency and memory depend on your hardware. Model names above are placeholders. No paid API or cloud account is required. The adapter connects to `127.0.0.1:11434` only; use local models, not cloud-backed models. Model provisioning may require internet access.

## Design

1. Validate supported file type and size; extract text per PDF page. Scans require external OCR.
2. Split into 180-word passages with 30-word overlap, preserving filename, page and content-based chunk ID. Identical file content is deduplicated (the last supplied filename is retained).
3. Build either a sparse TF-IDF index (word unigrams/bigrams) or normalized Ollama embeddings. Rank by cosine similarity, applying top-k and a configurable minimum score.
4. If there are no matching passages, abstain without calling a model. Otherwise show excerpts or send the retrieved evidence to the local chat model.
5. Require structured claims with source identifiers; validate their shape and that IDs reference retrieved evidence. Render plain text and expose the passages for inspection.

`rag.py` contains ingestion, retrieval and provider logic; `app.py` handles session-scoped UI; `cli.py` provides scripted access; `evaluate.py` produces a reproducible retrieval report.

An in-memory index avoids external database setup for a small demo corpus. There is no shared cross-user cache. Closing/clearing the session releases application references, but this is not a secure memory-erasure guarantee; Ollama can have its own retention/logging behavior. The app does not write uploaded documents to disk. Exported results contain source text.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluate.py
```

See `VERIFICATION.md` and `reports/retrieval.json`. The benchmark contains 9 relevant and 3 unrelated questions over three short synthetic documents: it is a smoke check, not an independent quality benchmark. Do not present its perfect score as real-world accuracy.

## Limits and next experiments

- Lexical retrieval is weak for paraphrases and cross-language questions. The dense alternative needs live evaluation on a representative corpus.
- A valid citation ID does not prove that a claim is supported. Prompt instructions reduce neither hallucinations nor prompt injection to zero. Human source review remains necessary.
- Similarity is not a confidence probability; the default 0.08 threshold is not calibrated. Relevant vocabulary can match an unanswerable question.
- No OCR, table reconstruction, conversational memory, reranking, authentication or durable vector database. Text/Markdown sources use page 1 as a placeholder.
- Intended for trusted local use: PDF limits are not a hardened parser sandbox. Bind to localhost; no public deployment is configured.
- Limits: 10 MB per file, 200 PDF pages, one million extracted characters per file, 20 files/30 MB total, 2,000 chunks per index, 2,000 question characters.

A useful next evaluation would use consented public documents, held-out answerable/unanswerable questions, page-level labels, lexical-vs-dense retrieval comparisons, groundedness review and latency measurements. Record model names and versions when doing that work.

## API references

- https://docs.ollama.com/api/chat
- https://docs.ollama.com/api/embed
- https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
