from typing import List, Tuple
import pandas as pd
import networkx as nx

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from datetime import datetime
from os import path as os_path

def unipartite_metapath_projection(data: Data,
                                   metapaths: List[List],
                                   edge_attr_func=lambda edge_index, metapath: 
                                        torch.ones(edge_index.shape[1],)*len(metapath)
                                   ) -> dict:

    new_data = dict()
    data_with_new_edges = T.AddMetaPaths(metapaths=metapaths)(data)

    def foo(i_metapath):
        i, metapath = i_metapath
        src_type = metapath[0][0]
        dst_type = metapath[-1][1]

        edge_index = data_with_new_edges[(src_type, f"metapath_{i}", dst_type)].edge_index

        new_data[i] = Data(
            x=None,
            edge_index=edge_index,
            # posso melhorar esse edge_attr
            edge_attr=edge_attr_func(edge_index, metapath)
        )

    for i, metapath in enumerate(metapaths):
        foo((i,metapath))

    return new_data


def networkx_unipartite_metapath_projection(**kwargs) -> dict:
    new_data = unipartite_metapath_projection(**kwargs)
    return {k: to_networkx(v) for k, v in new_data.items()}


def save_df(df: pd.DataFrame, filepath: str, filename: str = 'dataframe') -> None:
    save_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    df.to_csv(os_path.join(filepath, f"{save_timestamp}_{filename}_results.csv"))
    
    
class EarlyStopping:
    def __init__(self, patience: int = 5, threshold: float = 1e-4):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_state = None

    def step(self, model, loss: float):
        if loss < self.best_loss - self.threshold:
            self.best_state = model.state_dict()
            self.best_loss = loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience
