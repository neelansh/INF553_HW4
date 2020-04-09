import os
import sys
from pyspark import SparkContext, SparkConf
import json
import itertools
import math
import time
from queue import deque

appName = 'assignment4'
master = 'local[*]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


def create_adjacency_list(edges, vertices):
    adjacency_list = {}
    for (x, y) in edges:
        if(x in adjacency_list):
            adjacency_list[x].add(y)
        else:
            adjacency_list[x] = set([y])

        if(y in adjacency_list):
            adjacency_list[y].add(x)
        else:
            adjacency_list[y] = set([x])
    
    for v in set(vertices) - set(adjacency_list.keys()):
        adjacency_list[v] = set([])
            
    return adjacency_list

class Node:
    def __init__(self, idx, level, number_of_shortest_paths):
        self.id = idx
        self.level = level
        self.parent = []
        self.children = []
        self.number_of_shortest_paths = number_of_shortest_paths
        self.credit = 1
        return
    
    def __str__(self):
        return "{}, {}, {}".format(self.id, self.level, self.number_of_shortest_paths)
    
    def __repr__(self):
        return str(self)
    
class LevelIDMapper:
    def __init__(self):
        self.map = {}
        
    def update(self, level, node):
        if(level in self.map):
            self.map[level][node.id] = node
        else:
            self.map[level] = {node.id: node}
            
    def check_level(self, level, node_id):
        if(not level in self.map):
            return False
        
        return node_id in self.map[level]
    
    def get_node(self, level, node_id):
        
        if(not level in self.map):
            return False
        
        if(not node_id in self.map[level]):
            return False
        
        return self.map[level][node_id]
    
    
def BFS(start, adjacency_list):
    q = deque()
    start_node = Node(start, 0, 1)
    q.append(start_node)
    visited = set()
    visited.add(start)
    leaf_nodes = []
    level_node_map = LevelIDMapper()
    level_node_map.update(0, start_node)
    
    while(len(q) != 0):
        
        current_node = q.popleft()
        
        for child in adjacency_list[current_node.id]:
            if(child in visited and level_node_map.check_level(current_node.level+1, child)):
                child_node = level_node_map.get_node(current_node.level+1, child)
                child_node.parent.append(current_node)
                current_node.children.append(child_node)
                child_node.number_of_shortest_paths += current_node.number_of_shortest_paths
                level_node_map.update(current_node.level+1, child_node)
            if(not child in visited):
                child_node = Node(child, current_node.level+1, current_node.number_of_shortest_paths)
                child_node.parent.append(current_node)
                current_node.children.append(child_node)
                visited.add(child_node.id)
                q.append(child_node)
                level_node_map.update(current_node.level+1, child_node)
    
    
    return level_node_map, tuple(sorted(visited))
            

class EdgeCredit:
    def __init__(self):
        self.map = {}
    
    def update(self, edge, credit):
        edge = tuple(sorted(edge))
        if(edge in self.map):
            self.map[edge] += credit
        else:
            self.map[edge] = credit
            
    def divide_by_2(self):
        for edge in self.map.keys():
            self.map[edge] = self.map[edge] / 2.0
        return

def calculate_modularity(adjacency_list, communities, m):
    communities = list(communities)
    def helper(community):
        modularity = 0
        for (i, j) in itertools.combinations(community, 2):
            Aij = 1 if j in adjacency_list[i] else 0
            ki = len(adjacency_list[i])
            kj = len(adjacency_list[j])
            modularity += Aij-((ki*kj)/(2*m))
            
        return modularity
    
    modularity = sc.parallelize(communities).map(helper).reduce(lambda x, y: x+y)
        
            
    return modularity/(2*m)
            
    
    
def calculate_betweeness_and_communities(adjacency_list, vertices):
    ec = EdgeCredit()
    communities = set()
    for v in vertices:
        level_node_map, community = BFS(v, adjacency_list)
        communities.add(community)
        for level in sorted(level_node_map.map.keys(), reverse=True):
            for _, node in level_node_map.map[level].items():
                total_shortest_path = sum([parent.number_of_shortest_paths for parent in node.parent])
                for parent in node.parent:
                    credit = node.credit * parent.number_of_shortest_paths / total_shortest_path
                    parent.credit += credit
                    ec.update((node.id, parent.id), credit)
    
    ec.divide_by_2()   
    ec = sorted([(edge, credit)for edge, credit in ec.map.items()], key=lambda x: x[0][0])
    ec = sorted(ec, key=lambda x: -x[1])
    return ec, communities


def save_edge_credits(output_path, output):
    with open(output_path, 'wt') as f:
        for line in output:
            f.write(line)
    return

def save_output(output_path, output):
    output = sorted(sorted(list(output), key=lambda x: x[0]), key=lambda x: len(x))
    output = ["'"+"', '".join(x)+"'\n" for x in output]
    file = open(output_path, 'wt')
    for line in output:
        file.write(line)
        
    file.close()
    return


if __name__ == '__main__':
    st = time.time()
    input_path = sys.argv[1].strip()
    edges = sc.textFile(input_path).map(lambda x: tuple(sorted(x.split()))).collect()
    vertices = sc.textFile(input_path).flatMap(lambda x: x.split()).distinct().collect()
    original_number_of_edges = len(edges)
    original_adjacency_list = create_adjacency_list(edges, vertices)
    last_modularity = -3
    stopper_count = 0
    max_modularity = -sys.maxsize
    communities_with_max_modularity = None
    while len(edges) != 0:
        if(stopper_count > 15):
            break
        adjacency_list = create_adjacency_list(edges, vertices)
        edge_credits, communities = calculate_betweeness_and_communities(adjacency_list, vertices)
        modularity = calculate_modularity(original_adjacency_list, communities, original_number_of_edges)

        if(modularity > max_modularity):
    #         print(len(communities), modularity)
            max_modularity = modularity
            communities_with_max_modularity = communities
            stopper_count = 0

        if(modularity < last_modularity):
            stopper_count += 1

        if(original_number_of_edges == len(edges)):
            output = ["{}, {}\n".format(edge, credit) for edge, credit in edge_credits]
            save_edge_credits(sys.argv[2].strip(), output)

        last_modularity = modularity
        edges.remove(edge_credits[0][0])
    
    save_output(sys.argv[3].strip(), communities_with_max_modularity)
    print(time.time()-st)