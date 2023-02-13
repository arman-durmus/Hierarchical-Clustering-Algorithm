import random
from scipy import spatial
import numpy as np
import time
from itertools import chain

class TreeNode():
    """
    Node of a tree.

    Attributes
    ----------
    start: int
        The position of the first leaf node in the list.
    end: int
        The position of the last leaf node in the list+1.
    right: TreeNode
        The right child of the Node.
    left: TreeNode
        The left child of the Node.
    split: int
        The position of the leaf leftmost of the subtree rooted at the right child of the Node. 
    """
    def __init__(self, start, end) -> None:
        assert start < end
        self.start = start
        self.end = end
        self.right = None
        self.left = None
        self.split = None
    
    def __len__(self):
        """Return the number of leaves in the subtree of which this Node is the root."""
        return self.end - self.start
    
    def is_leaf(self):
        """Check whether the Node is a leaf."""
        return not (self.right or self.left)
    
    def __str__(self) -> str:
        if self.is_leaf(): return ""
        return f"{self.left.start}, {self.split}, {self.right.end}"
    
class Tree():
    """
    An implementation of the binary Tree Data Structure to help with swapping the points in the list according to the algorithm.

    Attributes
    ----------
    root: TreeNode
        The root Node of the Tree, representing the cluster containing all the Nodes.
    nodes: list of TreeNode
        The Nodes of the Tree.
    """
    def __init__(self, root):
        self.root = root
        self.nodes = [root]

    def split_node(self, node, i):
        """
        Splits a Node by creating two new children for the Node.

        Parameters
        ----------
        node: TreeNode
            A TreeNode.
        i: int
            Where to split the Node.
        Returns
        -------
        node.left, node.right: TreeNode
            the right and left child of the split Node
        """
        assert node.start < i
        assert node.end > i
        node.split = i
        node.left = TreeNode(node.start, i)
        node.right = TreeNode(i, node.end)
        self.nodes.append(node.right)
        self.nodes.append(node.left)
        return node.left, node.right

    def is_leaf(self, node):
        return not (node.right or node.left)

def make_tree(list_):
    """
    Create the Hierarchical clustering tree from the matrix resulting from the dynamic programming recurrence.

    Parameters
    ----------
    list_ : list
        A 2d list consisting of tuples of cost and split position, resulting from the dynamic programming algorithm.
    Returns
    -------
    tree: Tree
        The HC tree found by the Algorithm in a certain iteration.
    """
    root = TreeNode(0, len(list_)-1)
    tree = Tree(root)
    def helper(node):
        # a helper function to recursively construct the tree.
        if len(node) <= 1: return
        left, right = tree.split_node(node, list_[node.start][node.end][1])
        helper(left)
        helper(right)
    helper(root)
    return tree
    
def find_opt_tree(list_, sim):
    """
    Algorithm which finds the optimal HC tree for a list of data points in a set ordering.

    Parameters
    ----------
    list_ : list
        An ordered list of data points.
    sim: function
        the similarity function to be used, which takes two vectors/data points of the same dimensionality as input.
    Returns
    -------
    tree: Tree
        The HC tree found by the Algorithm in a certain iteration.
    opt_list[0][len(list_)][0] : int
        The cost of the found HC tree in a certain iteration. 
    """
    opt_list = [[0 for _ in range(len(list_) + 1)] for _ in range(len(list_) + 1)]
    cut_cost = [[[0 for _ in range(len(list_) + 1)] for _ in range(len(list_) + 1)] for _ in range(len(list_) + 1)]

    def find_opt(list_, start, end):
        nonlocal opt_list
        
        costs=[]
        cut_cost[start][start+1][end] = sum(sim(list_[start], b) for b in list_[start+1:end])
        costs.append((end-start)*cut_cost[start][start+1][end] + opt_list[start+1][end][0])
        for i in range(start+2, end):
            cut_cost[start][i][end] = cut_cost[start][i-1][end] + cut_cost[i-1][i][end] - cut_cost[start][i-1][i]
            costs.append((end-start)*cut_cost[start][i][end] + opt_list[start][i][0]  + opt_list[i][end][0])
        
        opt_list[start][end] = min(costs), (np.argmin(costs)+start+1)
        return opt_list[start][end]

    for i in range(0, len(list_)):
        opt_list[i][i+1] = (0, i)
    for i in range(2, len(list_)+1):
        for j in range(len(list_)+1-i):
            find_opt(list_, j, j+i)
    
    #construct the opimal HC tree by backtracking
    tree = make_tree(opt_list)
    
    return tree, opt_list[0][len(list_)][0]

def swap_inner_nodes(tree, list_):
    """
    Recursively swaps the right and left subtrees of inner Nodes of a given binary Tree with 50% probability for a given inner node.
    """
    def helper(node, list_):
        if node.is_leaf(): return list_
        if 0.5 > random.uniform(0,1):
            list_ = helper(node.right, list_)
            list_ = helper(node.left, list_)
        else:
            #print(f"Swap of {node.left.start}, {node.left.end}, {node.right.start}, {node.right.end}")
            list_ = helper(node.right, list_)
            list_ = helper(node.left, list_)
            
            list_ = list(chain(list_[:node.left.start], list_[node.right.start: node.right.end], list_[node.left.start: node.left.end], list_[node.right.end:]))
            node.left, node.right = node.right, node.left
        return list_
    return helper(tree.root, list_)

def main(list_, sim, iters=100, reports=25):
    """
    Executes the Algorithm for the given numer of iterations and creates reports with the given frequency.
    """
    oldcost = np.inf
    LEN = len(list_)
    report_list = []
    for iteration in range(iters):
        if len(list_) != LEN: break

        t1 = time.perf_counter()
        #find the optimal HC tree for the ordering

        tree, cost = find_opt_tree(list_, sim)
        print("Time for iteration ", iteration, "", time.perf_counter()-t1)
        res_list, res_tree = list_, tree
        print(str(tree.root))
        improvement = oldcost-cost
        print(f"Cost at iteration {iteration}: {cost}, Improvement: {improvement}")
        if iteration == 0:
            start_cost = cost
        if iteration+1%reports:
            report_list.append([iteration, cost, improvement])
        if iteration == iters-1:
            final_cost_improvement = [cost, improvement]
        if oldcost-cost < -0.001: break
        oldcost = cost

        #swap subtrees of innernodes with 50% possibility for each inner node. 
        list_ = swap_inner_nodes(tree, list_)


    print(f"Final cost: {cost}")
    return res_list, res_tree, start_cost, report_list, final_cost_improvement

if __name__ == "__main__":
    list_ = [(1, 2), (3, 4), (5, 6), (2,4), (1,5), (4,6)]
    def sim(a, b):
        """similarity function"""
        return 1 - spatial.distance.cosine(a, b)
    res_list, res_tree, start_cost, report_list, final_cost_improvement = main(list_, sim)
    print(res_list)
