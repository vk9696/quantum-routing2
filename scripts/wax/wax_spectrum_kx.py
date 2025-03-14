import networkx as nx
import matplotlib.pyplot as plt
import random
import csv
import logging
import numpy as np
import heapq
from scipy.spatial.distance import cdist
from itertools import islice

__all__ = [
    "create_network_graph",
    "f",
    "assign_noise_rate",
    "get_random_sd_pairs",
    "calculate_fidelity",
    "k_shortest_paths",
    "find_assigned_paths",
    "generate_sd_pairs",
    "compute_fidelity_for_all_paths",
    "save_to_csv"

]

n = 25 
width = 100
height = 100
alpha = 0.85
beta = 0.975
#s = 5 # number of sources
#d = 5 # number of destinations
wax_seed = 10
draw_network=False
fidelity_threshold = 0.53
X = 1

def create_network_graph(n, width, height, alpha, beta, s, d, wax_seed, draw_network):
    # Set the seed for random number generation
    if wax_seed is not None:
        random.seed(wax_seed)
        np.random.seed(wax_seed)
    
    G = nx.Graph()
    positions = {}
    max_distance = 0

    # Generate random nodes
    grid_nodes = []
    for i in range(n):
        x, y = random.randint(0, width), random.randint(0, height)
        G.add_node(i)
        grid_nodes.append(i)
        positions[i] = (x, y)

    # Calculate max_distance based on the actual node placements
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            if distance > max_distance:
                max_distance = distance
    #print("Max distance:", max_distance)

    # Add edges based on distance and probability
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            probability = beta * np.exp(-distance / (max_distance * alpha))
            if random.random() < probability:
                G.add_edge(i, j)
                
    # Generate unique nodes for sources and destinations
    available_nodes = list(G.nodes())
    np.random.shuffle(available_nodes)
    chosen_nodes = available_nodes[:s+d]
    source_nodes_ids = chosen_nodes[:s]
    destination_nodes_ids = chosen_nodes[s:]
    source_nodes = [n+i for i in range(s)]
    destination_nodes = [n+s+i for i in range(d)]
    
    # Attach sources and destinations to the network
    for source_node, node_id in zip(source_nodes, source_nodes_ids):
        G.add_node(source_node)
        G.add_edge(source_node, node_id)
    
    for destination_node, node_id in zip(destination_nodes, destination_nodes_ids):
        G.add_node(destination_node)
        G.add_edge(destination_node, node_id)

    # Check for connected components
    #if nx.number_connected_components(G) != 1:
        #print("Graph has more than one connected component, returning None.")
        #return None  # Or handle differently based on requirements
    
    pos = nx.spring_layout(G, seed=42)  # Use 2D spring layout

    if draw_network:
        plt.figure(figsize=(12, 8))  # Set the figure size

        # Draw the network
        nx.draw_networkx(G, pos, node_size=50, with_labels=True, font_size=8,
                         node_color=['yellow' if node in source_nodes else 'green' if node in destination_nodes else 'red' for node in G.nodes])

        plt.title("The Network")
        plt.show()

    return G, grid_nodes, source_nodes, destination_nodes

# Parameters
a = 10
LQ_eta = 0.8
HQ_eta = 0.999

# Function definition
def f(x, a=a):
    return np.exp(a * x) - np.exp(0.8 * a)

# Find the range of f(x, a) within the interval [LQ_eta, HQ_eta]
f_LQ = f(LQ_eta)
f_HQ = f(HQ_eta)

def assign_noise_rate(grid_nodes, source_nodes, destination_nodes, eff_seed):
    random.seed(eff_seed)  # Set the seed for reproducibility
    noise_rates = {}

    # Assign fixed high-quality noise rate to source nodes and destination nodes
    for node in source_nodes + destination_nodes:
        noise_rates[node] = HQ_eta

    # Assign random values from the range of f(x, a) and compute x analytically
    for node in grid_nodes:
        # Ensure the random value is always greater than min_random_value
        random_value = random.uniform(f_LQ, f_HQ)

        # Compute x using the provided analytical formula
        x_value = (np.log(random_value + np.exp(0.8 * a))) / a
        
        # Store the found x_value, rounded to 3 decimal places
        noise_rates[node] = round(x_value, 3)

    return noise_rates


def get_random_sd_pairs(source_nodes, destination_nodes, sd_pair_seed):
    sd_pairs = []

    # Create copies of the source and destination node lists
    available_source_nodes = source_nodes.copy()
    available_destination_nodes = destination_nodes.copy()

    # Shuffle the available source and destination node lists in unison with the provided seed
    random.seed(sd_pair_seed)
    random.shuffle(available_source_nodes)
    random.shuffle(available_destination_nodes)
    #logging.info(f"Fun get_random_sd_pairs shuffled source_nodes: {available_source_nodes}, shuffled destination_nodes: {available_destination_nodes}")

    # Generate pairs by pairing up the shuffled nodes
    for i in range(len(source_nodes)):
        source_node = available_source_nodes[i]
        destination_node = available_destination_nodes[i]
        sd_pairs.append((source_node, destination_node))

    return sd_pairs


def calculate_fidelity(noise_rates, path, threshold=fidelity_threshold):
    # Initialize variables to track the number of HQ and LQ nodes in the path
    product_term = 1

    # Set the initial fidelity constant
    F = 0.975

    # Count the number of HQ and LQ nodes in the path
    if path:
        for node in path:
            if node in noise_rates:
                eta = noise_rates[node]
                term = ((4 * (eta ** 2) - 1) / 3 ) * ((4 * F - 1) / 3)
                product_term *= term

        # Adjust the product term for the overall path
        fidelity = 0.25 * (1 + 3 * product_term * ((4 * F - 1) / 3))

        # Check if fidelity is less than the threshold
        if fidelity < threshold:
            fidelity = 'NA'
    
    else:
        fidelity = 'NA'

    return fidelity, product_term, F


def k_shortest_paths(G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )


def find_assigned_paths(sd_pairs, n_sd, eff_seed, fidelity_threshold, X):
    G, grid_nodes, source_nodes, destination_nodes = create_network_graph(n, width, height, alpha, beta, n_sd, n_sd, wax_seed, draw_network)
    noise_rates = assign_noise_rate(grid_nodes, source_nodes, destination_nodes, eff_seed)
    #for node, attrs in G.nodes(data=True):
        #logging.info(f"Fun find_shortest_paths weighted network: Node {node} has attributes {attrs}")
    
    best_paths = []
    assigned_paths = []
    path_order = []  # Store the order of shortest paths

    # Create a set to keep track of used edges
    used_edges = set()

    # Iterate over each pair and find the shortest path using Dijkstra's algorithm
    for pair in sd_pairs:
        source_node, destination_node = pair

        try:
            # Find the k-shortest path
            paths = k_shortest_paths(G, source_node, destination_node, k=50)
            path_fidelities = [(calculate_fidelity(noise_rates, path), path) for path in paths]

            # Filter paths with fidelity above the threshold
            valid_paths = [pf for pf in path_fidelities if pf[0][0] != 'NA' and pf[0][0] >= fidelity_threshold]

            if valid_paths:
                # Find the best path based on the shortest path length
                best_path = min(valid_paths, key=lambda x: len(x[1]))[1]
                best_paths.append(best_path)

                # Find the assigned path
                valid_paths_for_assigned = [pf for pf in valid_paths if len(pf[1]) <= len(best_path) + X]
                if valid_paths_for_assigned:
                    assigned_path = min(valid_paths_for_assigned, key=lambda x: x[0][0])[1]
                else:
                    assigned_path = []

                assigned_paths.append(assigned_path)

                # Mark the edges in the shortest path as used
                for i in range(len(assigned_path) - 1):
                    edge = (assigned_path[i], assigned_path[i + 1])
                    used_edges.add(edge)

                # Remove used edges from the graph
                G.remove_edges_from(used_edges)
            
            else:
                best_paths.append([])
                assigned_paths.append([])

            # Add the path order
            path_order.append(len(path_order) + 1)

        except nx.NetworkXNoPath:
            # Append an empty list if shortest path cannot be found
            best_paths.append([])
            assigned_paths.append([])
            path_order.append(len(path_order) + 1)

    return assigned_paths, path_order


def generate_sd_pairs(source_nodes, destination_nodes, num_combinations):
    sd_pairs_combinations = []
    for i in range(num_combinations):
        sd_pairs = get_random_sd_pairs(source_nodes, destination_nodes, i)
        sd_pairs_combinations.append(sd_pairs)
    return sd_pairs_combinations


def compute_fidelity_for_all_paths(sd_pairs_combinations, noise_rates, n_sd, eff_seed):
    all_paths_fidelity = []
    for sd_pairs in sd_pairs_combinations:
        shortest_paths, path_order = find_assigned_paths(sd_pairs, n_sd, eff_seed, fidelity_threshold, X)
        for order, path in zip(path_order, shortest_paths):  # Use zip to iterate through both path_order and shortest_paths simultaneously
            #logging.info(
                    #f"Fidelity calculation configuration: SD Pairs = {sd_pairs}, Path = {path}, \n"
                    #)
            fidelity, _, _ = calculate_fidelity(noise_rates, path)
            all_paths_fidelity.append((sd_pairs, path, len(path), order, fidelity))
    return all_paths_fidelity


def save_to_csv(all_paths_fidelity, n_sd, eff_seed):
    with open('data/wax_spectrum/k50x1_0.975.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for sd_pairs, path, path_length, path_order, fidelity in all_paths_fidelity:
            writer.writerow([n_sd, eff_seed, path, sd_pairs, path_length, path_order, fidelity])



def main():
    num_combinations = 100  # number of sd_pairs combinations

    with open('data/wax_spectrum/k50x1_0.975.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["n_sd", "eff_seed", "path", "sd_pairs", "path_length", "path_order", "fidelity"])

    for n_sd in {5}:

        # Create the network graph
        _ , grid_nodes, source_nodes, destination_nodes = create_network_graph(n, width, height, alpha, beta, n_sd, n_sd, wax_seed, draw_network)

        for eff_seed in range(1, 101, 1):
            logging.info(
                    f"Main configuration: n_sd = {n_sd}, eff_seed = {eff_seed} \n"
                    )
            # Assign noise rate to nodes
            noise_rates = assign_noise_rate(grid_nodes, source_nodes, destination_nodes, eff_seed)
            #logging.info(f"Main: noise rate: {noise_rates} \n")

            # Generate sd_pairs combinations
            sd_pairs_combinations = generate_sd_pairs(source_nodes, destination_nodes, num_combinations)
            #logging.info(f"Main: sd_pairs_combinations: {sd_pairs_combinations} \n")

            # Compute fidelity for all paths
            all_paths_fidelity = compute_fidelity_for_all_paths(sd_pairs_combinations, noise_rates, n_sd, eff_seed)

            # Save the data to a CSV file
            save_to_csv(all_paths_fidelity, n_sd, eff_seed)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
