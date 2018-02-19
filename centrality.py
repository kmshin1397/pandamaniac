import json
import random
import networkx as nx
import time
import sim
import numpy as np
from sklearn import preprocessing

input_filename = '2.10.31.json'
output_filename = 'ouput.txt'
num_seeds = input_filename.split('.')[1]
graph = None

def choose_nodes_betweenness(input_filename, num_seeds):
	global graph
	with open(input_filename) as f:
		graph = json.load(f)
		G = nx.Graph()
		degrees = {}
		for line in graph.items():
			node_id = line[0]
			neighbors = line[1]
			degrees[node_id] = len(neighbors)
			for neighbor in neighbors:
				G.add_edge(node_id, neighbor)
		print("Graph loaded")
		n = len(graph.items())
		betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
		print("Centralities calculated")
		sorted_between = sorted(betweenness.items(), key=lambda x: x[1])[-int(num_seeds):]
		result = []
		for i in sorted_between:
			result.append(i[0])
		#sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])

	return result

def choose_nodes(input_filename, num_seeds):
	random_nodes = random.sample(graph.keys(), int(num_seeds))

	return random_nodes

def op(input_filename, num_seeds):
	G = nx.Graph()
	degrees = {}
	clusters = {}
	print("Graph loaded")
	for line in graph.items():
		node_id = line[0]
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, neighbor)
	n = len(graph.items())
	#betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds)-10:]
	result = []
	for i in sorted_degrees[11:]:
		result.append(i[0])
	x = time.clock()
	print(len(result))
	max_node = '1'
	max_clustering = 9999
	for line in sorted_degrees[:10]:
		c = nx.algorithms.cluster.clustering(G, line[0])
		if (max_clustering > c):
			max_clustering = c
			max_node = line[0]
		#print(i+1)
	print(time.clock() - x)
	print("Centralities/clusters calculated")
	result.append(max_node)
	return result[:int(num_seeds)]

def fufu(input_filename, num_seeds, a, b):
	G = nx.Graph()
	degrees = []
	#print("Graph loaded")
	for line in graph.items():
		node_id = int(line[0])
		neighbors = line[1]
		degrees.append(float(len(neighbors)))
		for neighbor in neighbors:
			G.add_edge(node_id, int(neighbor))
		if (len(neighbors) ==0):
			G.add_node(node_id)

	# Normalize degrees for weighting with other heuristics
	degrees = np.array(degrees) / np.linalg.norm(degrees) 
	result = []
	# for i in sorted_degrees[11:]:
	# 	result.append(i[0])
	x = time.clock()
	clusters = []
	for node in range(len(graph)):
		c = nx.algorithms.cluster.clustering(G, node)
		clusters.append(c)

	clusters = np.array(clusters) / np.linalg.norm(clusters)
	#print(time.clock() - x)
	influence = {}
	for node, degree in enumerate(degrees):
		influence[node] = a*degree - b* clusters[node]

	result = sorted(influence.keys(), key=lambda x : influence[x])[-int(num_seeds):]
	return [str(i) for i in result]

def fufu2(input_filename, num_seeds):
	G = nx.Graph()
	degrees = {}
	clusters = {}
	print("Graph loaded")
	for line in graph.items():
		node_id = line[0]
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, neighbor)
	#n = len(graph.items())
	#betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds)/2:]
	result = []
	for i in sorted_degrees:
	 	result.append(i[0])
	x = time.clock()
	max_node = '1'
	max_clustering = 9999
	clusters = {}
	for line in graph.keys():
		c = nx.algorithms.cluster.clustering(G, line)
		if (max_clustering < c):
			max_clustering = c
			max_node = line
		clusters[line] = c
	#print(time.clock() - x)
	#print("Centralities/clusters calculated")
	sorted_clusters = sorted(clusters.items(), key=lambda x: x[1])[:int(num_seeds)*3]
	for i in sorted_clusters:
		result.append(i[0])	

	s = sorted(result, key=lambda x : degrees[x])[-int(num_seeds):]
	return s

def choose_nodes_degree(input_filename, num_seeds):
	G = nx.Graph()
	degrees = {}
	for line in graph.items():
		node_id = line[0]
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, neighbor)
	#print("Graph loaded")
	n = len(graph.items())
	#betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
	#print("Centralities calculated")
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds):]
	result = []
	for i in sorted_degrees:
		result.append(i[0])

	return result

def print_output(output_filename, chosen_nodes):
	with open(output_filename, 'w') as f:
		for i in range(50):
			for node in chosen_nodes:
				f.write(node)
				f.write('\n')

with open(input_filename) as f:
	graph = json.load(f)

max_op = None
max_score = 0
for i in range(20):
	x = float(i)/20 + 1.
	for j in range(20):
		y = float(j)/20
		a = fufu(input_filename, num_seeds, x, y)
		b = choose_nodes_degree(input_filename, num_seeds)
		seeds = {}
		seeds['op'] = a
		seeds['degrees'] = b
		r = sim.run(graph, seeds)
		if r['op'] > max_score:
			max_score = r['op']
			max_op = (x, y)

print(max_score)
print(max_op)
# print_output(output_filename, a)