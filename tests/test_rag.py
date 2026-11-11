import json
import unittest
from unittest.mock import Mock
from rag import Chunk, Index, Ollama, ingest, answer, validate_answer, ABSTENTION

class RagTests(unittest.TestCase):
    def test_chunk_coverage_and_overlap(self):
        chunks = ingest('a.txt', ' '.join(f'w{i}' for i in range(500)).encode(), 180, 30)
        self.assertEqual(chunks[0].text.split()[-30:], chunks[1].text.split()[:30])
        self.assertEqual({w for c in chunks for w in c.text.split()}, {f'w{i}' for i in range(500)})
    def test_duplicate_content(self):
        cs = ingest('a.txt', b'research application requirements')
        self.assertEqual(len(Index(cs + cs).chunks), 1)
    def test_unsupported_and_empty(self):
        for name, data in [('a.exe', b'abc'), ('a.txt', b''), ('a.txt', b'\xff')]:
            with self.assertRaises((ValueError, UnicodeError)): ingest(name, data)
    def test_bad_overlap(self):
        with self.assertRaises(ValueError): ingest('a.txt', b'abc', 5, 5)
    def test_retrieval_and_abstention(self):
        idx = Index(ingest('travel.txt', b'Borealis grants cover rail travel and accommodation.') + ingest('data.txt', b'Cirrus research datasets need provenance records.'))
        self.assertEqual(idx.search('rail travel')[0].chunk.source, 'travel.txt')
        self.assertTrue(answer(idx, 'quantum zebras')['abstained'])
    def test_empty_query(self):
        with self.assertRaises(ValueError): Index(ingest('a.txt', b'hello world')).search('')
    def test_valid_citation(self):
        self.assertEqual(validate_answer(json.dumps({'answerable': True, 'claims': [{'text':'Seven years.', 'sources':['S1']}]}), 1), 'Seven years. [S1]')
    def test_invalid_citation(self):
        for refs in [[], ['S9'], [1]]:
            with self.assertRaises(ValueError): validate_answer(json.dumps({'answerable': True,'claims':[{'text':'x', 'sources':refs}]}), 1)
    def test_no_evidence_no_generation(self):
        generator = Mock()
        result = answer(Index(ingest('a.txt', b'rail travel')), 'zebras', generator)
        generator.generate.assert_not_called()
        self.assertEqual(result['answer'], ABSTENTION)
    def test_provider_contract(self):
        provider = Ollama(chat_model='test')
        provider._post = Mock(return_value={'message': {'content': '{"answerable":true,"claims":[{"text":"Rail travel.","sources":["S1"]}]}'}})
        result = answer(Index(ingest('a.txt', b'rail travel')), 'rail', provider)
        self.assertIn('[S1]', result['answer'])
        self.assertEqual(provider._post.call_args.args[0], 'chat')
    def test_dense_normalization(self):
        provider = Ollama(embedding_model='test')
        provider._post = Mock(return_value={'embeddings': [[3,4]]})
        self.assertAlmostEqual(float((provider.embed(['x']) ** 2).sum()), 1)
    def test_pdf_empty(self):
        from pypdf import PdfWriter
        from io import BytesIO
        writer=PdfWriter(); writer.add_blank_page(width=200,height=200)
        stream=BytesIO();writer.write(stream)
        with self.assertRaisesRegex(ValueError, 'No text'): ingest('scan.pdf', stream.getvalue())

if __name__ == '__main__': unittest.main()
