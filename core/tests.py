from django.test import TestCase
from core import tools
import os


class ToolsTestCases(TestCase):

    def setUp(self):
        self.test_path = os.path.abspath('tests')

    def test_read_bad_beats(self):
        array_rr = tools.beat_reader("tests/badfiletest.beat")
        self.assertEqual(len(array_rr), 0)

    def test_read_good_beats(self):
        array_rr = tools.beat_reader("tests/goodfiletest.beat")
        self.assertEqual(len(array_rr), 100)

    def test_fill_rr_matrix(self):
        array_rr = tools.beat_reader("tests/goodfiletest.beat")
        matrix_rr = tools.fill_rr_matrix(array_rr)
        self.assertEqual(matrix_rr.shape, (tools.N_CELLS, tools.N_CELLS))
        s = matrix_rr.sum()
        self.assertEqual(s, 97)

    def test_log_rr_matrix(self):
        array_rr = tools.beat_reader("tests/goodfiletest.beat")
        matrix_rr = tools.fill_rr_matrix(array_rr)
        log_matrix_rr = tools.log_rr_matrix(matrix_rr, divider=1)
        s1 = int(log_matrix_rr.sum())
        self.assertEqual(s1, 24)

    def test_store_load_rr_matrix(self):
        array_rr = tools.beat_reader("tests/goodfiletest.beat")
        matrix_rr = tools.fill_rr_matrix(array_rr)
        log_matrix_rr = tools.log_rr_matrix(matrix_rr, divider=1)
        tools.store_matrix_list([log_matrix_rr], self.test_path)
        test_matrix_list = tools.load_matrix_list(self.test_path)
        self.assertEqual(len(test_matrix_list), 1)
        self.assertEqual(log_matrix_rr.sum(), test_matrix_list[0].sum())
        os.unlink(os.path.join(self.test_path, tools.MATRIX_LIST_FN))

    def test_store_load_heatmap(self):
        test_heatmap_fn = "heatmap.test"
        rr_matrix = tools.load_beat_matrix(os.path.join(self.test_path, "goodfiletest.beat"), min_beats=90)
        tools.store_heatmap(rr_matrix, self.test_path, test_heatmap_fn)
        rr_matrix_1 = tools.load_heatmap(self.test_path, test_heatmap_fn + ".npy")
        self.assertEqual(rr_matrix.sum(), rr_matrix_1.sum())
        os.unlink(os.path.join(self.test_path, test_heatmap_fn + ".npy"))

    def test_get_x_for_pca(self):
        rr_matrix = tools.load_beat_matrix(os.path.join(self.test_path, "goodfiletest.beat"), min_beats=90)
        x_for_pca = tools.get_x_for_pca([rr_matrix])
        self.assertEqual(len(x_for_pca), 1)
        self.assertEqual(int(x_for_pca[0].sum()), -16553)

