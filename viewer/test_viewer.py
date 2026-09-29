"""Viewer regression checks without models, API calls, or real datasets."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from .app import create_app
from .data import METRICS, DataError, Store, read_index
from .reports import benchmark_report


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('benchmark/test.jsonl', [
            {'id': 1, 'source': 'storystream', 'prompt': {'text': '<script>alert(1)</script>'}, 'golden_result': []},
            {'id': 2, 'source': 'wikipedia', 'prompt': {'text': 'two'}, 'golden_result': []},
        ])
        self.write('outputs/model/test.jsonl', [
            {'eval_set_id': 2, 'response_sf': [{'type': 'text', 'content': 'second response'}]},
            {'eval_set_id': 1, 'response_sf': [{'type': 'text', 'content': 'first response'}], 'response_tf': []},
        ])
        self.write('results/model/test.jsonl', [
            {'eval_set_id': 1, 'metrics': {'text_richness': 0, 'text_logicality': 1, 'context_fidelity': None}},
            {'eval_set_id': 2, 'metrics': {'text_richness': 1, 'context_fidelity': 0}},
            {'eval_set_id': 99, 'metrics': {'text_richness': 100}},
        ])
        self.app = create_app(self.root)
        self.app.testing = True
        self.client = self.app.test_client()

    def write(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_pages_and_id_join(self):
        for url in ['/', '/benchmark', '/response', '/report', '/report?view=categories']:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)
        html = self.client.get('/response?id=1').get_data(as_text=True)
        self.assertIn('first response', html)
        self.assertNotIn('second response', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>alert', html)
        self.assertIn('second response', self.client.get('/response?page=2').get_data(as_text=True))
        self.assertEqual(self.client.get('/benchmark?id=99').status_code, 400)
        self.assertEqual(self.client.get('/report?benchmark=missing').status_code, 404)
        html = self.client.get('/report').get_data(as_text=True)
        self.assertNotIn('n = ', html)
        self.assertNotIn('All benchmarks', html)
        self.assertIn('<td>50.00</td>', html)
        self.assertIn('<td class="incomplete">100.00</td>', html)
        self.assertIn('Red cells indicate incomplete evaluation', html)
        for url in ['/report?view=benchmarks', '/report.csv?view=benchmarks']:
            self.assertEqual(self.client.get(url).status_code, 400)

    def test_aggregation_and_coverage(self):
        row = benchmark_report(Store(self.root), 'test')[0]
        self.assertEqual((row['count'], row['expected'], row['extra']), (2, 2, 1))
        self.assertEqual(row['metrics']['text_richness'], {'score': 50, 'count': 2})
        self.assertEqual(row['metrics']['context_fidelity'], {'score': 0, 'count': 1})
        self.assertIsNone(row['metrics']['subject_consistency']['score'])
        self.assertEqual(row['categories']['knowledge']['score'], 50)
        self.assertEqual(row['categories']['knowledge']['valid']['context_fidelity'], 1)
        self.assertNotIn('subject_consistency', row['categories']['knowledge']['valid'])
        self.assertEqual(row['category_mean'], 50)
        self.assertEqual(row['sample_mean'], 50)

    def test_category_context_fidelity_coverage(self):
        for source, category in [('wikipedia', 'knowledge'), ('wikitravel', 'planning')]:
            with self.subTest(category=category):
                self.write('benchmark/test.jsonl', [{'id': 1, 'source': source}])
                metrics = {key: 1 for key in METRICS if key != 'subject_consistency'}
                metrics['context_fidelity'] = 0
                self.write('results/model/test.jsonl', [{'eval_set_id': 1, 'metrics': metrics}])
                entry = benchmark_report(Store(self.root), 'test')[0]['categories'][category]
                self.assertAlmostEqual(entry['score'], 500 / 6)
                self.assertFalse(entry['incomplete'])
                self.assertNotIn('subject_consistency', entry['valid'])
                metrics['context_fidelity'] = None
                self.write('results/model/test.jsonl', [{'eval_set_id': 1, 'metrics': metrics}])
                entry = benchmark_report(Store(self.root), 'test')[0]['categories'][category]
                self.assertTrue(entry['incomplete'])
                self.assertEqual(entry['valid']['context_fidelity'], 0)

    def test_csv(self):
        response = self.client.get('/report.csv')
        rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(rows[0]['context_fidelity'], '0.0')
        self.assertEqual(rows[0]['subject_consistency'], '')
        self.assertEqual(rows[0]['text_richness_valid'], '2')
        for view in ['categories']:
            self.assertEqual(self.client.get('/report.csv?view=' + view).status_code, 200)

    def test_duplicate_and_malformed_records(self):
        path = self.write('benchmark/bad.jsonl', [{'id': 1}, {'id': '1'}])
        with self.assertRaisesRegex(DataError, 'duplicate'):
            read_index(path, 'id')
        self.write('benchmark/bad.jsonl', [{'id': 1, 'metrics': []}])
        self.assertEqual(self.client.get('/benchmark?benchmark=bad').status_code, 400)

    def test_result_response_fallback(self):
        self.write('results/only/test.jsonl', [{'eval_set_id': 1, 'response': [{'type': 'text', 'content': 'legacy'}]}])
        self.assertEqual(Store(self.root).response('only', 'test', 1)['sf'][0]['content'], 'legacy')

    def test_image_access_boundaries(self):
        path = self.root / 'images/example.png'
        path.parent.mkdir()
        path.write_bytes(b'example')
        for value in ['images/example.png', str(path)]:
            response = self.client.get('/image', query_string={'path': value})
            self.assertEqual(response.status_code, 200)
            response.close()
        outside = self.root / 'private.png'
        outside.write_bytes(b'private')
        (path.parent / 'escape.png').symlink_to(outside)
        for value in ['private.png', 'images/../private.png', 'images/escape.png', '.env']:
            self.assertEqual(self.client.get('/image', query_string={'path': value}).status_code, 404)
        client = create_app(self.root, [self.root]).test_client()
        response = client.get('/image', query_string={'path': str(outside)})
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_empty_repository(self):
        client = create_app(self.root / 'empty').test_client()
        for url in ['/', '/response', '/report']:
            self.assertEqual(client.get(url).status_code, 200)


if __name__ == '__main__':
    unittest.main()
