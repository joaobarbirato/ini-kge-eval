import random
import numpy as np
import networkx as nx

from tqdm import tqdm
from typing import List

from src.utils import Data, unipartite_metapath_projection, to_networkx


def independent_cascade(G, seed_nodes, prob=0.2, max_iter=10, simulation: bool = True):
    """
    Simulates the Independent Cascade Model for diffusion.
    
    Parameters:
    - G: NetworkX graph
    - seed_nodes: List of initially activated nodes
    - prob: Activation probability for each edge
    - max_iter: Maximum number of iterations to simulate
    
    Returns:
    - activated_nodes: Set of nodes that were activated by the diffusion process
    """
    activated_nodes = set(seed_nodes)  # Initial activated nodes
    newly_activated = set(seed_nodes)  # Start with the seed nodes
    
    activated_nodes_per_iter = []
    
    # Step 3: Simulate the diffusion process
    for _ in range(max_iter):
        next_newly_activated = set()
        
        # Try to activate neighbors of newly activated nodes
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated_nodes:
                    # Each neighbor has a chance to be activated
                    # if random.random() < prob:
                    if random.random() < prob:
                        next_newly_activated.add(neighbor)
        
        # If no new nodes were activated, stop
        if not next_newly_activated:
            if not activated_nodes_per_iter:
                activated_nodes_per_iter.append(0)
            break

        # Add newly activated nodes to the activated set
        activated_nodes.update(next_newly_activated)
        newly_activated = next_newly_activated
        activated_nodes_per_iter.append(len(activated_nodes))
    
    return activated_nodes, activated_nodes_per_iter


def si_model(G, seed_nodes, prob=0.2, max_iter=10, simulation: bool = True):
    """
    Simulates the SI Model for diffusion.
    
    Parameters:
    - G: NetworkX graph
    - seed_nodes: List of initially activated nodes
    - prob: Activation probability for each edge
    - max_iter: Maximum number of iterations to simulate
    
    Returns:
    - activated_nodes: Set of nodes that were activated by the diffusion process
    """
    activated_nodes = set(seed_nodes)  # Initial activated nodes
    # newly_activated = set(seed_nodes)  # Start with the seed nodes
    
    activated_nodes_per_iter = []
    
    # Step 3: Simulate the diffusion process
    for _ in range(max_iter):
        newly_activated = set()
        
        # Try to activate neighbors of newly activated nodes
        # for node in newly_activated:
        for node in activated_nodes:
            for neighbor in G.neighbors(node):
                if random.random() < prob:  # Infection probability
                    newly_activated.add(neighbor)

            # for neighbor in G.neighbors(node):
            #     if neighbor not in activated_nodes:
            #         # Each neighbor has a chance to be activated
            #         if random.random() < prob:
            #             next_newly_activated.add(neighbor)
        
        # If no new nodes were activated, stop
        # if not next_newly_activated:
        #     break
        if len(activated_nodes) == len(G.nodes):  # Stop if all nodes are infected
            break

        # Add newly activated nodes to the activated set
        activated_nodes |= newly_activated
        # activated_nodes.update(next_newly_activated)
        # newly_activated = next_newly_activated
        activated_nodes_per_iter.append(len(activated_nodes))
    
    return activated_nodes, activated_nodes_per_iter


def sis_model(G, seed_nodes, prob=0.2, recovery_rate=0.1, max_iter=10, simulation: bool = True):
    """
    Simulates the SI Model for diffusion.
    
    Parameters:
    - G: NetworkX graph
    - seed_nodes: List of initially activated nodes
    - prob: Activation probability for each edge
    - max_iter: Maximum number of iterations to simulate
    
    Returns:
    - activated_nodes: Set of nodes that were activated by the diffusion process
    """
    activated_nodes = set(seed_nodes)  # Initial activated nodes
    # newly_activated = set(seed_nodes)  # Start with the seed nodes
    
    activated_nodes_per_iter = []
    
    # Step 3: Simulate the diffusion process
    for _ in range(max_iter):
        newly_activated = set()
        recovered_nodes = set()

        # Try to activate neighbors of newly activated nodes
        # for node in newly_activated:
        for node in activated_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in activated_nodes and random.random() < prob:  # Infection probability
                    newly_activated.add(neighbor)
                    
            if random.random() < recovery_rate:
                recovered_nodes.add(node)

        # In SIS/SIR, infected nodes can recover after some steps (recovery rate applied)            

        # If no new nodes were activated, stop
        if len(activated_nodes) == len(G.nodes):  # Stop if all nodes are infected
            break

        # Add newly activated nodes to the activated set
        activated_nodes |= newly_activated
        activated_nodes -= recovered_nodes

        # activated_nodes.update(next_newly_activated)
        # newly_activated = next_newly_activated
        activated_nodes_per_iter.append(len(activated_nodes))
    
    return activated_nodes, activated_nodes_per_iter


def sir_model(G, seed_nodes, prob=0.2, recovery_rate=0.1, max_iter=10, simulation: bool = True):
    """
    Simulates the SIR Model for diffusion.
    
    Parameters:
    - G: NetworkX graph
    - seed_nodes: List of initially activated nodes
    - prob: Activation probability for each edge
    - max_iter: Maximum number of iterations to simulate
    
    Returns:
    - activated_nodes: Set of nodes that were activated by the diffusion process
    """
    activated_nodes = set(seed_nodes)  # Initial activated nodes
    recovered_nodes = set()
    # newly_activated = set(seed_nodes)  # Start with the seed nodes
    
    activated_nodes_per_iter = []
    
    # Step 3: Simulate the diffusion process
    for _ in range(max_iter):
        newly_activated = set()
        newly_recovered_nodes = set()
        # Try to activate neighbors of newly activated nodes
        # for node in newly_activated:
        for node in activated_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in activated_nodes and random.random() < prob:  # Infection probability
                    newly_activated.add(neighbor)
                    
            if random.random() < recovery_rate:
                newly_recovered_nodes.add(node)

        if len(activated_nodes) == len(G.nodes):  # Stop if all nodes are infected
            break

        activated_nodes |= newly_activated
        recovered_nodes |= newly_recovered_nodes
        
        activated_nodes -= recovered_nodes
        activated_nodes_per_iter.append(len(activated_nodes))
    
    return activated_nodes, activated_nodes_per_iter


def meta_sir_model(data: Data, 
                   seed_nodes, 
                   metapath: List[str],
                   prob=0.2, 
                   recovery_rate=0.1, 
                   max_iter=10, 
                   simulation: bool = True):
    
    G = to_networkx(unipartite_metapath_projection(data, [metapath])[0])

    activated_nodes = set(seed_nodes)  # Initial activated nodes
    recovered_nodes = set()
    # newly_activated = set(seed_nodes)  # Start with the seed nodes
    
    activated_nodes_per_iter = []
    
    # Step 3: Simulate the diffusion process
    for _ in range(max_iter):
        newly_activated = set()
        newly_recovered_nodes = set()
        # Try to activate neighbors of newly activated nodes
        for node in activated_nodes:            
            for neighbor in G.neighbors(node):
                if neighbor not in activated_nodes and random.random() < prob:  # Infection probability
                    newly_activated.add(neighbor)
                    
            if random.random() < recovery_rate:
                newly_recovered_nodes.add(node)

        if len(activated_nodes) == len(G.nodes):  # Stop if all nodes are infected
            break

        activated_nodes |= newly_activated
        recovered_nodes |= newly_recovered_nodes
        
        activated_nodes -= recovered_nodes
        activated_nodes_per_iter.append(len(activated_nodes))
    
    return activated_nodes, activated_nodes_per_iter


def run_diffusion_simulations(
        num_simulations, 
        diffusion_model: callable = independent_cascade,  # type: ignore
        n_jobs: int = -1,
        verbose=True, 
        **kwargs):
    
    def _equalize_list_size(list_of_lists: List[list]):
        if len(list_of_lists) < 2:
            return len(list_of_lists[0])
        
        max_len = max(*list(map(lambda x: len(x), list_of_lists)))
        for i in range(len(list_of_lists)):
            list_of_lists[i] += [list_of_lists[i][-1]] * (max_len - len(list_of_lists[i]))
        return max_len

    def _simulate():
        activated_nodes, dni_per_epoch = diffusion_model(**kwargs)
        return len(activated_nodes), dni_per_epoch
    
    # with Parallel(n_jobs=n_jobs) as para:
    #     results = para(delayed(_simulate)() for _ in tqdm(range(num_simulations)))
    
    # all_activated_nodes, all_dni_per_epoch = zip(*results)
    
    all_activated_nodes = []
    all_dni_per_epoch = []
    
    for _ in tqdm(range(num_simulations), disable=not verbose):
        # Randomly select `num_seeds` nodes as the initial seeds
        # seed_nodes = random.sample(list(G.nodes), num_seeds)
        
        # Run the diffusion model
        activated_nodes, dni_per_epoch = diffusion_model(simulation=num_simulations>1, **kwargs)
        all_activated_nodes.append(len(activated_nodes))
        all_dni_per_epoch.append(dni_per_epoch)
    
    max_len = _equalize_list_size(all_dni_per_epoch) 
    
    return np.mean(all_activated_nodes), np.std(all_activated_nodes), np.mean(np.array(all_dni_per_epoch), axis=0), np.std(np.array(all_dni_per_epoch), axis=0), max_len
