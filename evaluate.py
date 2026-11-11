"""Deterministic retrieval smoke benchmark; NOT an LLM or legal accuracy evaluation."""
import json
from pathlib import Path
from rag import Index, ingest
CASES = [
 ('Aurora records retained years', 'aurora.md'),
 ('Aurora technical proposal budget', 'aurora.md'),
 ('Aurora submission deadline', 'aurora.md'),
 ('Borealis rail travel accommodation', 'borealis.md'),
 ('Borealis maximum visit duration', 'borealis.md'),
 ('Borealis appeals notification', 'borealis.md'),
 ('Cirrus source code installation', 'cirrus.md'),
 ('Cirrus datasets provenance dictionary', 'cirrus.md'),
 ('Cirrus ethics participant data', 'cirrus.md'),
 ('quantum zebra teleportation', None),
 ('pizza mozzarella oven', None),
 ('volcano magma tectonics', None),
]
def main():
 root=Path(__file__).parent
 chunks=[c for p in sorted((root/'data/demo').glob('*.md')) for c in ingest(p.name,p.read_bytes())]
 index=Index(chunks); rows=[]
 for question,expected in CASES:
  hits=index.search(question)
  actual=hits[0].chunk.source if hits else None
  rows.append({'question':question,'expected':expected,'actual':actual,'correct':expected==actual})
 positives=[r for r in rows if r['expected']];negatives=[r for r in rows if r['expected'] is None]
 report={'scope':'Synthetic lexical smoke benchmark, 3 short documents. Not independent held-out or LLM quality evaluation.',
 'retrieval':'TF-IDF word unigrams/bigrams, cosine, threshold 0.08',
 'hit_at_1':sum(r['correct'] for r in positives)/len(positives),
 'unrelated_query_abstention':sum(r['correct'] for r in negatives)/len(negatives),'cases':rows}
 (root/'reports').mkdir(exist_ok=True)
 (root/'reports/retrieval.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
