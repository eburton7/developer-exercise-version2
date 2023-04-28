
# Please create a simple example use of the pynn library for your end user. Assume that the end
# user knows a lot about their subject matter but has only a basic understanding of Python.

# Meaningful examples may include reading a file, finding a few nearby points and writing them
# out to the console.

""" 
below code is a simple example for the usage of the Nearest Neighbor Index
"""
from nearest_neighbor_index import NearestNeighborIndex

# define a list of 2D points
points = [(1, 1), (2, 3), (4, 2), (5, 6), (6, 1)]

# create an instance of the NearestNeighborIndex class
nni = NearestNeighborIndex(points)

# define a query point
query_point = (3, 4)

# find the nearest point to the query point using the find_nearest method
nearest_point = nni.find_nearest(query_point)

print(f"The nearest point to {query_point} is {nearest_point}")
