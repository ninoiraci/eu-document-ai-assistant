import unittest
from io import BytesIO
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from rag import ingest, Index

class PdfTests(unittest.TestCase):
    def test_page_provenance(self):
        writer = PdfWriter()
        for text in ['Aurora retains records for seven years.', 'Borealis visits last ninety days.']:
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        chunks = ingest('example.pdf', output.getvalue())
        self.assertEqual([c.page for c in chunks], [1, 2])
        result = Index(chunks).search('Borealis visits')
        self.assertEqual(result[0].chunk.page, 2)
        self.assertEqual(result[0].chunk.source, 'example.pdf')
