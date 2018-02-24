import json
import random
import networkx as nx
import time
import sim
import numpy as np
from networkx.algorithms import approximation as apxa
from networkx.algorithms import community
from sklearn import preprocessing

input_filename = '6.10.7.json'
output_filename = 'ouput.txt'
num_seeds = input_filename.split('.')[1]
graph = None

def choose_nodes_closeness(input_filename, num_seeds):
	x = time.clock()
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
	betweenness = nx.algorithms.centrality.closeness_centrality(G)
	print("Centralities calculated")
	sorted_between = sorted(betweenness.items(), key=lambda x: x[1])[-int(num_seeds):]
	result = []
	for i in sorted_between:
		result.append(i[0])
	#sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])
	print(time.clock() - x)
	return result

def choose_nodes_load_centrality(input_filename, num_seeds):
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
	betweenness = nx.algorithms.centrality.load_centrality(G)
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

# Take the least clustered nodes among the top 20 by closeness centrality
def strategy_1(input_filename, num_seeds):
	G = nx.Graph()
	print("Graph loaded")
	for line in graph.items():
		node_id = int(line[0])
		neighbors = line[1]
		for neighbor in neighbors:
			G.add_edge(node_id, int(neighbor))
		if (len(neighbors) ==0):
			G.add_node(node_id)

	n = len(graph.items())
	degrees = nx.algorithms.centrality.closeness_centrality(G)
	#betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds)-10:]
	result = []
	for i in sorted_degrees:
		result.append(i[0])
	x = time.clock()
	clusters = []
	for node in range(len(graph)):
		c = nx.algorithms.cluster.clustering(G, node)
		clusters.append(c)
		#print(i+1)
	print(time.clock() - x)
	print("Centralities/clusters calculated")
	return [str(i) for  i in sorted(result, key=lambda x: clusters[int(x)])[:int(num_seeds)]]

# Weighted average of various centralities
def strategy_2(input_filename, num_seeds):
	x = time.clock()
	G = nx.Graph()
	degrees = []
	Gconnected = nx.Graph()
	#print("Graph loaded")
	for line in graph.items():
		node_id = int(line[0])
		neighbors = line[1]
		degrees.append(float(len(neighbors)))
		for neighbor in neighbors:
			G.add_edge(node_id, int(neighbor))
			Gconnected.add_edge(node_id, int(neighbor))
		if (len(neighbors) ==0):
			G.add_node(node_id)

	dominating = apxa.min_weighted_dominating_set(G)
	#dominating = max(nx.connected_components(Gconnected), key= lambda x:len(x))
	complement = set(G.nodes()) - dominating
	print("1")
	centralities = nx.algorithms.centrality.closeness_centrality(G)
	#centralities = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(dominating - vc))
	centrality = []
	for node in dominating:
		centrality.append(centralities[node])

	# Normalize degrees for weighting with other heuristics
	centrality = np.array(centrality) / np.linalg.norm(centrality) 
	print("2")
	centralities2 = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(complement))
	centrality2 = []
	for node in dominating:
		centrality2.append(centralities2[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality2 = np.array(centrality2) / np.linalg.norm(centrality2) 
	print("3")
	centralities3 = nx.algorithms.centrality.subgraph_centrality(G)
	centrality3 = []
	for node in dominating:
		centrality3.append(centralities2[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality3 = np.array(centrality2) / np.linalg.norm(centrality2) 
	print("4")
	result = []
	# for i in sorted_degrees[11:]:
	# 	result.append(i[0])
	clusters = []
	for node in dominating:
		c = nx.algorithms.cluster.clustering(G, node)
		clusters.append(c)

	clusters = np.array(clusters) / np.linalg.norm(clusters)
	print("5")
	influence = {}
	for node, degree in enumerate(centrality):
		influence[node] = 5*degree + 5*centrality2[node] + 5*centrality3[node] - 1.* clusters[node]

	result = sorted(influence.keys(), key=lambda x : influence[x])[-int(num_seeds):]
	print(time.clock() - x)
	dominating = list(dominating)
	return [str(dominating[i]) for i in result]

# Helper function taking weighted average for influence measure incorporating node neighbors
def calc_centrality(G, node_id, c1, c2, c3, clusters, subset):
	neighbors = G[node_id]
	#centralities = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(dominating - vc))
	centrality = []
	for node in neighbors:
		if (node in subset):
			centrality.append(c1[node])

	# Normalize degrees for weighting with other heuristics
	centrality = sum(centrality) #/ len(centrality) 

	centrality2 = []
	for node in neighbors:
		if (node in subset):
			centrality2.append(c2[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality2 = sum(centrality2) #/ len(centrality2) 

	centrality3 = []
	for node in neighbors:
		if (node in subset):
			centrality3.append(c3[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality3 = sum(centrality3) #/ len(centrality3) 

	cluster = []
	for node in neighbors:
		if node in subset:
			cluster.append(clusters[node_id])
	cluster = sum(cluster)

	result = -1*(2*c1[node_id] - c2[node_id] + 3*c3[node_id]) + 2*(2*centrality + centrality2 + 2.4*centrality3) - 5*cluster
	#result = 1*(c1[node_id] + c2[node_id]) + 2*(centrality + centrality2) - 5*cluster
	# print(centrality2)
	# print(clusters[node_id])
	# print('______________')
	#result = 2*c1[node_id] + 2*c3[node_id] + 2*c2[node_id] + 2.5*centrality + 2.5*centrality2 + 2.5*centrality3 - 1.*clusters[node_id]
	return result

# Testing different weights
def calc_centrality2(G, node_id, c1, c2, c3, clusters, subset):
	neighbors = G[node_id]
	#centralities = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(dominating - vc))
	centrality = []
	for node in neighbors:
		if (node in subset):
			centrality.append(c1[node])

	# Normalize degrees for weighting with other heuristics
	centrality = sum(centrality) #/ len(centrality) 

	centrality2 = []
	for node in neighbors:
		if (node in subset):
			centrality2.append(c2[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality2 = sum(centrality2) #/ len(centrality2) 

	centrality3 = []
	for node in neighbors:
		if (node in subset):
			centrality3.append(c3[node])
	
	# Normalize degrees for weighting with other heuristics
	centrality3 = sum(centrality3) #/ len(centrality3) 

	cluster = []
	for node in neighbors:
		if node in subset:
			cluster.append(clusters[node_id])
	cluster = sum(cluster)

	result = 1*(2*c1[node_id] + 10*c2[node_id] + 3*c3[node_id]) + 2*(2*centrality + 1.5*centrality2 + 2.4*centrality3) - 5*cluster
	#result = 1*(c1[node_id] + c2[node_id]) + 2*(centrality + centrality2) - 5*cluster
	# print(centrality2)
	# print(clusters[node_id])
	# print('______________')
	#result = 2*c1[node_id] + 2*c3[node_id] + 2*c2[node_id] + 2.5*centrality + 2.5*centrality2 + 2.5*centrality3 - 1.*clusters[node_id]
	return result

# Combine different centrality measures, incorporating neighbors, using 1st set of weightings
def strategy_3(input_filename, num_seeds):
	x = time.clock()
	G = nx.Graph()
	degrees = {}
	Gconnected = nx.Graph()
	#print("Graph loaded")
	for line in graph.items():
		node_id = int(line[0])
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, int(neighbor))
			Gconnected.add_edge(node_id, int(neighbor))
		if (len(neighbors) ==0):
			G.add_node(node_id)
 	
	#dominating = apxa.min_weighted_dominating_set(G)
	#dominating = max(nx.connected_components(Gconnected), key= lambda x:len(x))
	#complement = set(G.nodes()) - dominating
	#print(nx.number_of_nodes(G))
	#dominating = set(random.sample(G.nodes(), .9*nx.number_of_nodes(G)))
	dominating = set(G.nodes())
	print("Dominating done")
	ha = time.clock()
	centralities = nx.algorithms.centrality.subgraph_centrality(G)
	print("subgraph done: " + str(time.clock()-ha))
	whoa = time.clock()
	#centralities2 = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(complement))
	#centralities2 = nx.algorithms.centrality.closeness_centrality(G)
	centralities2 = nx.algorithms.centrality.degree_centrality(G)
	print("Degree done:" + str(time.clock()-whoa))
	#centralities3 = nx.algorithms.centrality.harmonic_centrality(G, nbunch=dominating)
	ay = time.clock()
	centralities3 = nx.algorithms.centrality.eigenvector_centrality(G, max_iter=150, tol=1.0*10**-6)
	print("Harmonic done: " + str(time.clock() - ay))

	welp = time.clock()
	clusters = {}
	for node in dominating:
		c = nx.algorithms.cluster.clustering(G, node)
		clusters[node] = c

	print("Cluster done: " + str(time.clock()- welp))
	lol = time.clock()
	influence = {}
	for node in dominating:
		influence[node] = calc_centrality(G, node, centralities, centralities2, centralities3, clusters, dominating)
	print("Neighbors: " + str(time.clock()-lol))
	# Normalize degrees for weighting with other heuristics
	# centrality = np.array(centrality) / np.linalg.norm(centrality) 

	sorted_inf = sorted(influence.keys(), key=lambda x : influence[x])
	result = sorted_inf[-int(num_seeds):]
	# degrees = nx.algorithms.centrality.degree_centrality(G)
	# max_degrees = sorted(degrees.keys(), key=lambda x: degrees[x])[-int(num_seeds)+a:]
	# count = 1
	# done = False
	# while(not done):
	# 	changed = False
	# 	for node in result:
	# 		if node in max_degrees:
	# 			result.remove(node)
	# 			result.append(sorted_inf[-int(num_seeds)-count])
	# 			count += 1
	# 			changed = True
	# 	if not changed: 
	# 		done = True

	print(time.clock() - x)
	# dominating = list(dominating)
	# max_degrees.extend(result[-a:])
	return [str(i) for i in result]

# Use 2nd set of weights
def strategy_4(input_filename, num_seeds):
	x = time.clock()
	G = nx.Graph()
	degrees = {}
	Gconnected = nx.Graph()
	#print("Graph loaded")
	for line in graph.items():
		node_id = int(line[0])
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, int(neighbor))
			Gconnected.add_edge(node_id, int(neighbor))
		if (len(neighbors) ==0):
			G.add_node(node_id)
 	
	dominating = apxa.min_weighted_dominating_set(G)
	#dominating = max(nx.connected_components(Gconnected), key= lambda x:len(x))
	complement = set(G.nodes()) - dominating
	#print(nx.number_of_nodes(G))
	#dominating = set(random.sample(G.nodes(), .9*nx.number_of_nodes(G)))
	#dominating = set(G.nodes())
	print("Dominating done")
	ha = time.clock()
	centralities = nx.algorithms.centrality.subgraph_centrality(G)
	print("subgraph done: " + str(time.clock()-ha))
	whoa = time.clock()
	centralities2 = nx.algorithms.centrality.betweenness_centrality_subset(G, list(dominating), list(complement))
	#centralities2 = nx.algorithms.centrality.closeness_centrality(G)
	#centralities2 = nx.algorithms.centrality.degree_centrality(G)
	print("Degree done:" + str(time.clock()-whoa))
	#centralities3 = nx.algorithms.centrality.harmonic_centrality(G, nbunch=dominating)
	ay = time.clock()
	centralities3 = nx.algorithms.centrality.eigenvector_centrality(G, max_iter=150, tol=1.0*10**-6)
	print("Harmonic done: " + str(time.clock() - ay))

	welp = time.clock()
	clusters = {}
	for node in dominating:
		c = nx.algorithms.cluster.clustering(G, node)
		clusters[node] = c

	print("Cluster done: " + str(time.clock()- welp))
	lol = time.clock()
	influence = {}
	for node in dominating:
		influence[node] = calc_centrality2(G, node, centralities, centralities2, centralities3, clusters, dominating)
	print("Neighbors: " + str(time.clock()-lol))
	# Normalize degrees for weighting with other heuristics
	# centrality = np.array(centrality) / np.linalg.norm(centrality) 

	sorted_inf = sorted(influence.keys(), key=lambda x : influence[x])
	result = sorted_inf[-int(num_seeds):]
	# degrees = nx.algorithms.centrality.degree_centrality(G)
	# max_degrees = sorted(degrees.keys(), key=lambda x: degrees[x])[-int(num_seeds)+a:]
	# count = 1
	# done = False
	# while(not done):
	# 	changed = False
	# 	for node in result:
	# 		if node in max_degrees:
	# 			result.remove(node)
	# 			result.append(sorted_inf[-int(num_seeds)-count])
	# 			count += 1
	# 			changed = True
	# 	if not changed: 
	# 		done = True

	print(time.clock() - x)
	# dominating = list(dominating)
	# max_degrees.extend(result[-a:])
	return [str(i) for i in result]

# For testing new networkx methods
def kcomponents(input_filename, num_seeds, degrees):
	# G = nx.Graph()
	# degrees = {}
	# clusters = {}
	# print("Graph loaded")
	# for line in graph.items():
	# 	node_id = line[0]
	# 	neighbors = line[1]
	# 	degrees[node_id] = len(neighbors)
	# 	for neighbor in neighbors:
	# 		G.add_edge(node_id, neighbor)	
	# start = time.clock()
	# dominating = community.asyn_fluidc(G, 3)
	# for i in dominating:
	# 	print(i)
	# print(time.clock()-start)
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds):]
	result = []
	for i in sorted_degrees:
		result.append(i[0])

	return [str(i) for i in result]

# Just take top by degree
def choose_nodes_degree(input_filename, num_seeds):
	t = time.clock()
	G = nx.Graph()
	degrees = {}
	for line in graph.items():
		node_id = line[0]
		neighbors = line[1]
		degrees[node_id] = len(neighbors)
		for neighbor in neighbors:
			G.add_edge(node_id, neighbor)
	#print("Graph loaded")
	#betweenness = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
	#print("Centralities calculated")
	sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])[-int(num_seeds):]
	result = []
	for i in sorted_degrees:
		result.append(i[0])

	print(time.clock()-t)
	return result

# Output nodes to file
def print_output(output_filename, chosen_nodes):
	with open(output_filename, 'w') as f:
		for i in range(50):
			for node in chosen_nodes:
				f.write(node)
				f.write('\n')

with open(input_filename) as f:
	graph = json.load(f)

# a = None
# print(len(graph))
# if len(graph) < 2000:
# 	a = fufu(input_filename, num_seeds)
# else:
# 	a = fufu2(input_filename, num_seeds)
# # max_op = None
# # max_score = 0
# # for i in range(20):
# # 	x = float(i)/20 + 1.
# # 	for j in range(20):
# # 		y = float(j)/20
		
# # 		a = fufu(input_filename, num_seeds, x, y)
# # 		b = choose_nodes_degree(input_filename, num_seeds)
# # 		seeds = {}
# # 		seeds['op'] = a
# # 		seeds['degrees'] = b
# # 		r = sim.run(graph, seeds)
# # 		if r['op'] > max_score:
# # 			max_score = r['op']
# # 			max_op = (x, y)
# print(a)
b = strategy_3(input_filename, num_seeds)
# print(b)
# seeds = {}
# seeds['op'] = a
# seeds['degrees'] = b
# r = sim.run(graph, seeds)
# print(r)

print_output(output_filename, b)