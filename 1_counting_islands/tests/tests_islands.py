import unittest

from main import count_islands


class TestCountIslands(unittest.TestCase):

    def test_example_1(self):
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 1],
        ]

        self.assertEqual(count_islands(grid, 3, 3), 2)

    def test_example_2(self):
        grid = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]

        self.assertEqual(count_islands(grid, 3, 4), 3)

    def test_example_3(self):
        grid = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ]

        self.assertEqual(count_islands(grid, 3, 4), 2)

    def test_empty_ocean(self):
        grid = [
            [0, 0],
            [0, 0],
        ]

        self.assertEqual(count_islands(grid, 2, 2), 0)

    def test_single_island(self):
        grid = [
            [1, 1],
            [1, 1],
        ]

        self.assertEqual(count_islands(grid, 2, 2), 1)

    def test_diagonal_cells_not_connected(self):
        grid = [
            [1, 0],
            [0, 1],
        ]

        self.assertEqual(count_islands(grid, 2, 2), 2)

    def test_single_row(self):
        grid = [
            [1, 0, 1, 1, 0, 1]
        ]

        self.assertEqual(count_islands(grid, 1, 6), 3)

    def test_single_column(self):
        grid = [
            [1],
            [0],
            [1],
            [1],
        ]

        self.assertEqual(count_islands(grid, 4, 1), 2)

    def test_one_cell_ocean(self):
        grid = [
            [0]
        ]

        self.assertEqual(count_islands(grid, 1, 1), 0)

    def test_one_cell_island(self):
        grid = [
            [1]
        ]

        self.assertEqual(count_islands(grid, 1, 1), 1)

    def test_large_island(self):
        grid = [[1] * 100 for _ in range(100)]

        self.assertEqual(count_islands(grid, 100, 100), 1)


if __name__ == "__main__":
    unittest.main()