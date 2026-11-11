"""Command-line entry point; run python cli.py --help."""
import argparse
import json
from pathlib import Path
from rag import Index, Ollama, ingest, answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--documents', type=Path, default=Path(__file__).parent / 'data/demo')
    parser.add_argument('--chat-model', default='')
    parser.add_argument('--embedding-model', default='')
    parser.add_argument('--threshold', type=float, default=.08)
    parser.add_argument('--top-k', type=int, default=4)
    args = parser.parse_args()
    try:
        files = sorted(p for p in args.documents.iterdir() if p.is_file() and p.suffix.lower() in {'.pdf', '.txt', '.md'})
        if len(files) > 20 or sum(p.stat().st_size for p in files) > 30 * 1024 * 1024:
            raise ValueError('Maximum 20 files and 30 MB total.')
        chunks = [c for p in files for c in ingest(p.name, p.read_bytes())]
        index = Index(chunks, Ollama(embedding_model=args.embedding_model) if args.embedding_model else None)
        result = answer(index, args.question, Ollama(chat_model=args.chat_model) if args.chat_model else None,
                        args.top_k, args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
