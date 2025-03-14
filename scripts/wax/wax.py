import networkx as nx
import matplotlib.pyplot as plt
import random
import csv
import logging
import numpy as np
import heapq
from scipy.spatial.distance import cdist

__all__ = [
    "create_network_graph",
    "assign_noise_rate",
    "create_network_with_weighted_index",
    "get_random_sd_pairs",
    "vertex_weighted_dijkstra_path",
    "calculate_fidelity",
    "find_shortest_paths",
    "generate_sd_pairs",
    "compute_fidelity_for_all_paths",
    "save_to_csv"

]

n = 25 
width = 100
height = 100
alpha = 0.85
beta = 0.775
#s = 5 # number of sources
#d = 5 # number of destinations
wax_seed = 10
draw_network=False

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

HQ_eta = 0.999
LQ_eta = 0.8
def assign_noise_rate(grid_nodes, source_nodes, destination_nodes, HQ_percent, HQ_seed):
    random.seed(HQ_seed)
    noise_rates = {}

    # Assign noise rate to source nodes and destination nodes
    for node in source_nodes + destination_nodes:
        noise_rates[node] = HQ_eta

    # Calculate the number of high-quality nodes based on the HQ_percent
    num_hq_nodes = int(len(grid_nodes) * (HQ_percent / 100))

    # Assign noise rate to grid nodes randomly
    hq_nodes = random.sample(grid_nodes, num_hq_nodes)
    #logging.info(f"Fun Assign_noise_rate hq_nodes: {hq_nodes} \n")
    for node in grid_nodes:
        if node in hq_nodes:
            noise_rates[node] = HQ_eta  # High quality node
        else:
            noise_rates[node] = LQ_eta  # Low quality node
    
    return noise_rates


def create_network_with_weighted_index(noise_rates, network, high_quality_weight=1, low_quality_weight=1):
    weighted_network = network.copy()

    for node, noise_rate in noise_rates.items():
        weight = high_quality_weight if noise_rate == 0.999 else low_quality_weight
        #logging.info(f"Fun in for loop create_network_with_weighted_index weight: {weight}, noise_rate: {noise_rate}")
        weighted_network.nodes[node]["weight"] = weight
    return weighted_network


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


def vertex_weighted_dijkstra_path(G, source, target, weight="weight"):
    """Returns the shortest weighted path from source to target in G, using vertex weights.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node

    target : node
        Ending node

    weight : string or function
        If this is a string, then vertex weights will be accessed via the
        node attribute with this key (that is, the weight of the vertex `v`
        will be ``G.nodes[v][weight]``). If no such vertex attribute exists,
        the weight of the vertex is assumed to be one.

    Returns
    -------
    path : list
        List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.
    """
    # Initialize the priority queue with source
    queue = [(0, source, [])]
    visited = set()

    while queue:
        (cost, current, path) = heapq.heappop(queue)

        # If this node has already been visited, skip it
        if current in visited:
            continue

        # Add this node to the visited set
        visited.add(current)

        # Build the new path
        path = path + [current]

        # If we have reached the target, return the path
        if current == target:
            return path

        # Otherwise, add all neighbors to the priority queue
        for neighbor in G[current]:
            if neighbor not in visited:
                # Compute the vertex weight for the neighbor
                vertex_weight = float(G.nodes[neighbor].get(weight, 1))
                heapq.heappush(queue, (cost + vertex_weight, neighbor, path))

    # If there is no path, raise an exception
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


def calculate_fidelity(noise_rates, path, threshold=0.53):
    # Initialize variables to track the number of HQ and LQ nodes in the path
    num_hq_nodes = 0
    num_lq_nodes = 0

    # Count the number of HQ and LQ nodes in the path
    if path:
        for node in path:
            if node in noise_rates:
                if noise_rates[node] == HQ_eta:
                    num_hq_nodes += 1
                elif noise_rates[node] == LQ_eta:
                    num_lq_nodes += 1

        # Set the initial fidelity constant
        F = 0.975

        # Calculate the fidelity
        fidelity = ((1/4) * (1 + 3 * (((4 * (HQ_eta ** 2) - 1) / 3) ** (num_hq_nodes)) *
                            (((4 * (LQ_eta ** 2) - 1) / 3) ** (num_lq_nodes)) *
                            (((4 * F - 1) / 3) ** (num_hq_nodes + num_lq_nodes + 1))))

        # Check if fidelity is less than the threshold
        if fidelity < threshold:
            fidelity = 'NA'
    
    else:
        fidelity = 'NA'

    return fidelity, num_hq_nodes, num_lq_nodes


def find_shortest_paths(sd_pairs, n_sd, HQ_percent, HQ_seed):
    network, grid_nodes, source_nodes, destination_nodes = create_network_graph(n, width, height, alpha, beta, n_sd, n_sd, wax_seed, draw_network)
    noise_rates = assign_noise_rate(grid_nodes, source_nodes, destination_nodes, HQ_percent, HQ_seed)
    G = create_network_with_weighted_index(noise_rates, network)
    #for node, attrs in G.nodes(data=True):
        #logging.info(f"Fun find_shortest_paths weighted network: Node {node} has attributes {attrs}")
    
    shortest_paths = []
    path_order = []  # Store the order of shortest paths

    # Create a set to keep track of used edges
    used_edges = set()

    # Iterate over each pair and find the shortest path using Dijkstra's algorithm
    for pair in sd_pairs:
        source_node, destination_node = pair

        try:
            # Find the shortest path
            shortest_path = vertex_weighted_dijkstra_path(G, source_node, destination_node)

            # Calculate the fidelity of the found path
            fidelity, _, _ = calculate_fidelity(noise_rates, shortest_path)

            if fidelity != 'NA':

                # Append the shortest path to the list
                shortest_paths.append(shortest_path)

                # Mark the edges in the shortest path as used
                for i in range(len(shortest_path) - 1):
                    edge = (shortest_path[i], shortest_path[i + 1])
                    used_edges.add(edge)

                # Remove used edges from the graph
                G.remove_edges_from(used_edges)
            
            else:
                shortest_paths.append([])

            # Add the path order
            path_order.append(len(path_order) + 1)

        except nx.NetworkXNoPath:
            # Append an empty list if shortest path cannot be found
            shortest_paths.append([])
            path_order.append(len(path_order) + 1)

    return shortest_paths, path_order


def generate_sd_pairs(source_nodes, destination_nodes, num_combinations):
    sd_pairs_combinations = []
    for i in range(num_combinations):
        sd_pairs = get_random_sd_pairs(source_nodes, destination_nodes, i)
        sd_pairs_combinations.append(sd_pairs)
    return sd_pairs_combinations


def compute_fidelity_for_all_paths(sd_pairs_combinations, noise_rates, n_sd, HQ_percent, HQ_seed):
    all_paths_fidelity = []
    for sd_pairs in sd_pairs_combinations:
        shortest_paths, path_order = find_shortest_paths(sd_pairs, n_sd, HQ_percent, HQ_seed)
        for order, path in zip(path_order, shortest_paths):  # Use zip to iterate through both path_order and shortest_paths simultaneously
            #logging.info(
                    #f"Fidelity calculation configuration: SD Pairs = {sd_pairs}, Path = {path}, \n"
                    #)
            fidelity, _, _ = calculate_fidelity(noise_rates, path)
            all_paths_fidelity.append((sd_pairs, path, len(path), order, fidelity))
    return all_paths_fidelity


def save_to_csv(all_paths_fidelity, n_sd, HQ_seed, HQ_percent):
    with open('data/wax/beta_0.775/sp_nsd_5.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for sd_pairs, path, path_length, path_order, fidelity in all_paths_fidelity:
            writer.writerow([n_sd, HQ_percent, HQ_seed, path, sd_pairs, path_length, path_order, fidelity])



def main():
    num_combinations = 100  # number of sd_pairs combinations

    with open('data/wax/beta_0.775/sp_nsd_5.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["n_sd", "HQ_percent", "HQ_seed", "path", "sd_pairs", "path_length", "path_order", "fidelity"])

    for n_sd in {5}:

        # Create the network graph
        _ , grid_nodes, source_nodes, destination_nodes = create_network_graph(n, width, height, alpha, beta, n_sd, n_sd, wax_seed, draw_network)

        for HQ_percent in range(48, 101, 4):
        #for HQ_percent in {20}:
            for HQ_seed in range(1, 101, 1):
            #for HQ_seed in {5}:
                logging.info(
                        f"Main configuration: n_sd = {n_sd}, HQ_percent = {HQ_percent}, HQ_seed = {HQ_seed} \n"
                        )
                # Assign noise rate to nodes
                noise_rates = assign_noise_rate(grid_nodes, source_nodes, destination_nodes, HQ_percent, HQ_seed)
                #logging.info(f"Main: noise rate: {noise_rates} \n")

                # Generate sd_pairs combinations
                sd_pairs_combinations = generate_sd_pairs(source_nodes, destination_nodes, num_combinations)
                #logging.info(f"Main: sd_pairs_combinations: {sd_pairs_combinations} \n")

                # Compute fidelity for all paths
                all_paths_fidelity = compute_fidelity_for_all_paths(sd_pairs_combinations, noise_rates, n_sd, HQ_percent, HQ_seed)

                # Save the data to a CSV file
                save_to_csv(all_paths_fidelity, n_sd, HQ_seed, HQ_percent)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
