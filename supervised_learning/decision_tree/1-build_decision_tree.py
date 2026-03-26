#!/usr/bin/env python3
'''module documented'''
import numpy as np


class Node:
    '''Node class documented'''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        '''init documented'''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''method documented'''
        if (self.right_child.max_depth_below() >
                self.left_child.max_depth_below()):
            max_d = self.right_child
        else:
            max_d = self.left_child
        return max_d.max_depth_below()

    def count_nodes_below(self, only_leaves=False):
        '''method documented'''
        count = 0
        if only_leaves:
            if self.is_leaf:
                return 1
            count += self.left_child.count_nodes_below(True)
            count += self.right_child.count_nodes_below(True)
        else:
            count += 1
            count += self.left_child.count_nodes_below(False)
            count += self.right_child.count_nodes_below(False)
        return count


class Leaf(Node):
    '''Leaf class documented'''
    def __init__(self, value, depth=None):
        '''init documented'''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        '''method documented'''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''method documented'''
        return 1


class Decision_Tree():
    '''decision_tree class documented'''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''init documented'''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        '''method documented'''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''method documented'''
        return self.root.count_nodes_below(only_leaves=only_leaves)
