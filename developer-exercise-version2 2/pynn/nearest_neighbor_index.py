import math
from collections import deque

"""
    This code has improved find_nearest_fast method by using k-d tree. 
    The k-d tree is a binary tree that is used to partition a k-dimensional space into small regions.  
    
    ** quick overview/understanding of Nearest Neighbor was found on;
    -- https://en.wikipedia.org/wiki/Nearest_neighbor_search
    """

class Node:
    """
    A Node in a k-d tree.
    """
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    """
    A k-d tree that stores 2-dimensional points.
    """
    def __init__(self, points):
        self.root = self.build_kd_tree(points, axis=0)

    def build_kd_tree(self, points, axis):
        if not points:
            return None
        points = sorted(points, key=lambda x: x[axis])
        mid = len(points) // 2
        return Node(points[mid], axis,
                    left=self.build_kd_tree(points[:mid], (axis + 1) % 2),
                    right=self.build_kd_tree(points[mid + 1:], (axis + 1) % 2))

    def find_nearest_neighbor(self, query_point):
        """
        Returns the point in the tree that is closest to the given query point.
        """
        if not self.root:
            return None
        best_node = None
        best_dist = float('inf')
        to_visit = deque([(self.root, query_point, 0)])
        while to_visit:
            node, point, axis = to_visit.pop()
            dist = self.distance(point, node.point)
            if dist < best_dist:
                best_node = node
                best_dist = dist
            if node.left and point[axis] <= node.point[axis] + best_dist:
                to_visit.append((node.left, point, (axis + 1) % 2))
            if node.right and point[axis] >= node.point[axis] - best_dist:
                to_visit.append((node.right, point, (axis + 1) % 2))
        return best_node.point

    def distance(self, p1, p2):
        """
        Returns the Euclidean distance between points p1 and p2.
        """
        deltax = p1[0] - p2[0]
        deltay = p1[1] - p2[1]
        return math.sqrt(deltax * deltax + deltay * deltay)

class NearestNeighborIndex:
    """
    NearestNeighborIndex is intended to index a set of provided points to provide fast nearest
    neighbor lookup.
    """

    def __init__(self, points):
        """
        takes an array of 2d tuples as input points to be indexed.
        """
        self.points = points
        self.kd_tree = KDTree(points)

    @staticmethod
    def find_nearest_slow(query_point, haystack):
        """
        find_nearest_slow returns the point that is closest to query_point. If there are no indexed
        points, None is returned.
        
        *** this algorithm iterates through each point in the given haystack. Calculates the euclidean distance
        between the 'query_point' and the point from the haystack. 
        """

        min_dist = None
        min_point = None

        for point in haystack:
            deltax = point[0] - query_point[0]
            deltay = point[1] - query_point[1]
            dist = math.sqrt(deltax * deltax + deltay * deltay)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_point = point

        return min_point

    def find_nearest_fast(self, query_point):
        """
        This algorithm uses kd tree. This method recursively partitions the points 
        into two halves based on their coordinates, creates binary search tree based on those partitions,
        query point is then traversed down the tree to find the nearest neighbor. 
    
        Based off information from different sources such as: 
        -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
        -- https://en.wikipedia.org/wiki/K-d_tree
        
        find_nearest_fast returns the point that is closest to query_point. If there are no indexed
        points, None is returned.
        
        Returns the point in the index that is closest to the given query point.
        """
        return self.kd_tree.find_nearest_neighbor(query_point)

    def find_nearest(self, query_point):
        """
        purpose of this method is to allow the code to be easily modified to switch between the slow
        and fast implementations, w/o changing the external interface of the nearestneighborindex class.
        """
        return self.find_nearest_fast(query_point)
