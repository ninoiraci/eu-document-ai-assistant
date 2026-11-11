# Verification record

Executed on 2026-10-29, macOS ARM64, Python 3.13. Re-run on 2026-11-11 after translating the interface to English (same results).

- 14 automated tests passed (`reports/tests.txt`). Includes Streamlit AppTest for demo indexing, query, result and clear; real in-memory two-page PDF extraction and page provenance; overlap, deduplication, malformed inputs, irrelevant-query abstention, citation validation, mocked chat contract and embedding normalization.
- `python evaluate.py`: 9/9 relevant synthetic queries retrieved the expected document first; 3/3 unrelated queries produced no retrieval. All questions are simple and share vocabulary with a tiny synthetic corpus. These figures do not estimate real-world accuracy.
- CLI executed on the demo corpus; JSON result saved in `reports/example-answer.json`.
- UI tested through Streamlit's AppTest, not a manual browser visual review.
- Actual Ollama inference was not run: executable unavailable on the development machine. Dense retrieval integration and generated answer quality remain unverified with real models. Unit mocks do not replace this validation.

Direct package versions used: numpy 2.4.1, scikit-learn 1.8.0, Streamlit 1.66.1, pypdf 6.21.4. `requirements.txt` supplies compatible ranges; `requirements-tested.txt` pins these four direct dependencies. Neither file locks transitive dependencies.
