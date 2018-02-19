import json
import random

input_filename = '8.35.1.json'
output_filename = 'ouput.txt'
num_seeds = 35

def choose_nodes(input_filename, num_seeds):
	with open(input_filename) as f:
		graph = json.load(f)
		n = len(graph)
		random_nodes = random.choices(list(graph.keys()), k=num_seeds)

	return random_nodes

def print_output(output_filename, chosen_nodes):
	with open(output_filename, 'w') as f:
		for i in range(50):
			for node in chosen_nodes:
				f.write(node)
				f.write('\n')

rand = choose_nodes(input_filename, num_seeds)
print_output(output_filename, rand)


