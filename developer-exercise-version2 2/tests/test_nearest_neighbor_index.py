"""nn_search_test"""

import random
import time
import unittest

from pynn import NearestNeighborIndex


class NearestNeighborIndexTest(unittest.TestCase):
    def test_basic(self):
        """
        test_basic tests a handful of nearest neighbor queries to make sure they return the right
        result.
        """

        test_points = [
            (1, 2),
            (1, 0),
            (10, 5),
            (-1000, 20),
            (3.14159, 42),
            (42, 3.14159),
        ]

        uut = NearestNeighborIndex(test_points)

        self.assertEqual((1, 0), uut.find_nearest((0, 0)))
        self.assertEqual((-1000, 20), uut.find_nearest((-2000, 0)))
        self.assertEqual((42, 3.14159), uut.find_nearest((40, 3)))

    def test_benchmark(self):
        """
        test_benchmark tests a bunch of values using the slow and fast version of the index
        to determine the effective speedup.
        """

        def rand_point():
            return (random.uniform(-1000, 1000), random.uniform(-1000, 1000))

        index_points = [rand_point() for _ in range(10000)]
        query_points = [rand_point() for _ in range(1000)]

        expected = []
        actual = []

        # Run the baseline slow tests to get the expected values.
        start = time.time()
        for query_point in query_points:
            expected.append(NearestNeighborIndex.find_nearest_slow(query_point, index_points))
        slow_time = time.time() - start

        # don't include the indexing time when benchmarking
        uut = NearestNeighborIndex(index_points)

        # Run the indexed tests
        start = time.time()
        for query_point in query_points:
            actual.append(uut.find_nearest(query_point))
        new_time = time.time() - start

        print(f"slow time: {slow_time:0.2f}sec")
        print(f"new time: {new_time:0.2f}sec")
        print(f"speedup: {(slow_time / new_time):0.2f}x")

    # TODO: Add more test cases to ensure your index works in different scenarios
    def test_empty_points(self):
        """
        test_empty_points tests that if an empty list of points is provided,
        find_nearest() should return None.
        """

        uut = NearestNeighborIndex([])

        self.assertIsNone(uut.find_nearest((0, 0)))

    def test_duplicate_points(self):
        """
        test_duplicate_points tests that find_nearest() returns one of the points
        with the same minimum distance if there are multiple points with the same distance
        to the query point.
        """

        test_points = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
        ]

        uut = NearestNeighborIndex(test_points)

        self.assertIn(uut.find_nearest((0, 0)), test_points[:2])

    def test_identical_points(self):
        """
        test_identical_points tests the behavior of the index when there are multiple points in the index
        that have the same coordinates as the query point.
        """

        test_points = [
            (1, 2),
            (1, 0),
            (10, 5),
            (-1000, 20),
            (3.14159, 42),
            (42, 3.14159),
            (1, 2),  # Add a point with the same coordinates as the first point
            (3.14159, 42),  # Add a point with the same coordinates as the fifth point
        ]

        uut = NearestNeighborIndex(test_points)

        # Query a point that has the same coordinates as the first point in the index
        self.assertEqual((1, 2), uut.find_nearest((1, 2)))

        # Query a point that has the same coordinates as the fifth point in the index
        self.assertEqual((3.14159, 42), uut.find_nearest((3.14, 43)))

        # Query a point that has the same coordinates as the seventh point in the index
        self.assertEqual((1, 2), uut.find_nearest((0, 1)))
