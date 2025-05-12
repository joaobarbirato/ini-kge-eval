from pandas import Series
from typing import List
from joblib import Parallel, delayed
from functools import partial
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm

from .diffusion import run_diffusion_simulations, sir_model


def monotonicity(R: Series) -> float:
    def _get_ties_in_same_rank(r: int, R: Series) -> int:
        ties = len(R[R == r])
        return ties

    S = 0
    for r in R:
        ties = _get_ties_in_same_rank(r, R)
        S += ties * (ties - 1) / (len(R) * (len(R) - 1))

    return (1 - S) ** 2


def embedding_sim(G: nx.Graph, embeddings: torch.Tensor, node_type: str = 'author', verbose=True):
    measure = {}
    node2index, index2node = {}, {}
    i = 0
    for n, d in G.nodes(data=True):
        if d['type'] == node_type:
            node2index[n] = i
            index2node[i] = n
            i += 1
    
    dists = torch.cdist(
        torch.index_select(embeddings, 0, torch.tensor(list(index2node.keys()))), 
        torch.index_select(embeddings, 0, torch.tensor(list(index2node.keys())))
    )
    norm_dists = (dists - dists.min())/(dists.max() - dists.min())
    simis = (1 - norm_dists)
    score = torch.sum( simis, dim=0 )
    for i in tqdm(range(dists.shape[0]), disable=not verbose):
        measure[index2node[i]] = score[i].item()
    return measure


def sort_by_measure(measures: dict):
    return list(map(lambda x: x[0], 
                    sorted(measures.items(), 
                           key=lambda t: t[1], 
                           reverse=True)))


def sir_measure(G: nx.graph, 
                node_type: str, 
                prob: float = 0.01, 
                recovery_rate: float = 0.005, 
                num_simulations: int = 1000):
    
    measure = {}
    # avg dni per node
    _run = partial(
        run_diffusion_simulations,
        num_simulations=num_simulations,
        diffusion_model=sir_model,
        verbose=False,
        n_jobs=-1,
        # kwargs
        G=G,
        prob=prob,
        recovery_rate=recovery_rate
    )
    def foo(n):
        measure[n] = _run(seed_nodes=[n])[0]
        
    with Parallel(n_jobs=16, require='sharedmem') as para:
        para(delayed(foo)(n) for n, d
             in tqdm(G.nodes(data=True),
                     total=G.number_of_nodes(),
                     desc="sir measure")
             if d['type'] == node_type)
    return measure


def centrality_measure(G: nx.Graph, node_type: str):
    measure = {}
    measure['eig'] = nx.eigenvector_centrality(G=G, max_iter=10000)
    print("eig done")
    measure['deg'] = nx.degree_centrality(G=G)
    print("deg done")
    measure['btwn'] = nx.betweenness_centrality(G=G)
    print("btwn done")
    measure['clos'] = nx.closeness_centrality(G=G)
    print("clos done")
    measure['coreness'] = nx.core_number(G=G)
    print("coreness done")

    for key in measure.keys():
        measure[key] = {node: measure[key][node]
            for node, data
            in G.nodes(data=True)
            if data['type'] == node_type}

    return measure


def nlc(G: nx.Graph, embeddings: torch.Tensor, node_type: str = 'author', enabled_types: List[str] = [], verbose: bool =True):
    nlc = {}
    node2index, index2node = {}, {}
    i = 0
    for n, d in G.nodes(data=True):
        if (enabled_types and d['type'] in enabled_types) or (not enabled_types):
            node2index[n] = i
            index2node[i] = n
            i += 1
            
    print("\tnumber of nodes:", G.number_of_nodes())
    coreness = nx.core_number(G=G)
    def _nlc(n):
        ck = coreness[n]
        nlc[n] = 0
        n_idx = node2index[n]
        for neigh in G.neighbors(n):
            neigh_idx = node2index[neigh]
            repr_norm = torch.linalg.norm(embeddings[n_idx] - embeddings[neigh_idx])
            nlc[n] += ck * torch.exp(-repr_norm).item()
        
    with Parallel(n_jobs=16, require='sharedmem') as para:
        para(delayed(_nlc)(n) for n, d
             in tqdm(G.nodes(data=True),
                     total=G.number_of_nodes(),
                     desc="nlc measure", 
                     disable=not verbose)
             if d['type'] == node_type)
    
    return nlc


def mahe_node_relevancy(G: nx.Graph, embeddings: torch.Tensor, node_type: str = 'author', verbose:bool = True):
    mahe = {}
    node2index, index2node = {}, {}
    i = 0
    for n, d in G.nodes(data=True):
        if d['type'] == node_type:
            node2index[n] = i
            index2node[i] = n
            i += 1
    
    # cossine similarity to rank nodes
    normalized_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cossim_matrix = torch.matmul(embeddings, normalized_embeddings.T) ## the most similar nodes
    cossim_scores = torch.sum(cossim_matrix, dim=0)
    
    for n, d in tqdm(G.nodes(data=True), total=G.number_of_nodes(), disable=not verbose):
        if d['type'] == node_type:            
            mahe[n] = cossim_scores[node2index[n]].item()

    return mahe


# NYI
def mkni(G: nx.Graph, embeddings: torch.Tensor, node_type: str = 'author', verbose: bool = True):
    measure = {}
    di_measures = {}
    ii_measures = {}

    node2index, index2node = {}, {}
    i = 0
    # for i, (n, d) in enumerate(G.nodes(data=True)):
    for n, d in G.nodes(data=True):
        if d['type'] == node_type:
            node2index[n] = i
            index2node[i] = n
            i += 1
    
    normalized_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cossim_matrix = torch.matmul(embeddings, normalized_embeddings.T)
    
    def _direct_influence(i):
        di = 0
        for j in nx.neighbors(G, i):    # TODO: j must be of the same type as i
                                        # which doesn't make sense in the context of
                                        # the metapath2vec implementation
            if G._node[j]['type'] != G._node[i]['type']:
                continue
            
            csim = cossim_matrix[node2index[i], node2index[j]].item()
            s = torch.sum([cossim_matrix[node2index[j], node2index[k]]
                           for k in nx.neighbors(G, j) 
                           if G.node[k]['type'] == G.node[i]['type']]).item()
            di += csim / s
        return di
    
    def _indirect_influence(i): # NYI
        ii = 0
        # TODO: The weights of the connected edges correspond 
        # to the number of edges they form based on the 
        # intermediate P
        """
        for j in nx.neighbors(G, i):
            csim = cossim_matrix[node2index[i], node2index[j]].item()
            s = torch.sum([cossim_matrix[node2index[j], node2index[k]] 
                           for k in nx.neighbors(G, j)]).item()
            ii += (csim * np.exp(1 - nx.clustering(G, i))) / s
        """
        # for n, d in tqdm(G.nodes(data=True), total=G.number_of_nodes(), disable=not verbose):
        #    if d['type'] == node_type:
        #        cc = nx.clustering(G, n)

        return ii
    
    def _normalize_influence_dict(influence_dict):
        max_val = max(influence_dict.values())
        min_val = min(influence_dict.values())
        for k, v in influence_dict.items():
            influence_dict[k] = (v - min_val) / (max_val - min_val)
        return influence_dict

    for n, d in tqdm(G.nodes(data=True), total=G.number_of_nodes(), disable=not verbose):
        if d['type'] == node_type:            
            di_measures[n] = _direct_influence(i)
            ii_measures[n] = _indirect_influence(i)
    
    for n, d in tqdm(G.nodes(data=True), total=G.number_of_nodes(), disable=not verbose):
        if d['type'] == node_type:            
            measure[n]  =   _normalize_influence_dict(di_measures)[n] + \
                            _normalize_influence_dict(ii_measures)[n]
    

    return measure
    
    