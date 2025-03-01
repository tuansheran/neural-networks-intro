import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


nodes = [
    {"name": "Alice", "class": "A", "dating": False},
    {"name": "Bob", "class": "B", "dating": True},
    {"name": "Charlie", "class": "A", "dating": False},
    {"name": "Diana", "class": "B", "dating": True},
    {"name": "Eve", "class": "C", "dating": False},
]

edges = [(0, 1), (0, 2), (1, 3), (3, 4), (2, 4)]
