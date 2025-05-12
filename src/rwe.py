from typing import List
import torch
import torch_geometric as pyg
from torch_geometric.nn import MetaPath2Vec
from tqdm import tqdm

from src.utils import EarlyStopping


def train_mp2vec(
        model: MetaPath2Vec,
        loader: pyg.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        log_steps: int = 100, 
        max_epochs: int = 20,
        patience: int = 5,
        early_stopping_thr: float = 1e-4,
        verbose: bool = False) -> List[float]:

    train_losses = []
    early_stopper = EarlyStopping(patience=patience, threshold=early_stopping_thr)
    for epoch in range(1, max_epochs+1):
        model.train()

        cumulative_loss = 0
        log_loss = 0
        
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch: {epoch}", disable=not verbose)
        for i, (pos_rw, neg_rw) in pbar:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Update parameters
            optimizer.step()
            cumulative_loss += loss.item()
            log_loss += loss.item()
            if (i + 1) % log_steps == 0:
                pbar.set_postfix(step=f'{i + 1:05d}/{len(loader)}',
                                 loss=f'{log_loss / log_steps:.4f}')
                log_loss = 0
        
        avg_loss = cumulative_loss / len(loader)
        train_losses.append(avg_loss)

        scheduler.step(avg_loss)
        early_stopper.step(model, avg_loss)
        
        # Early-stopping
        if early_stopper.should_stop():
            pbar.set_postfix(step=f'{i + 1:05d}/{len(loader)}',
                             loss=f'{avg_loss:.4f}',
                             early_stopped='True')
            break
        
    if early_stopper.best_state is not None:
        with torch.no_grad():
            model.load_state_dict(early_stopper.best_state)

    return train_losses


def setup_metapath2vec(
        data: pyg.data.HeteroData,
        metapath: List[tuple],
        embedding_dim: int = 64,
        walk_length: int = 50,
        walks_per_node: int = 5,
        num_negative_samples: int = 5,
        ) -> MetaPath2Vec:

    model = MetaPath2Vec(
        data.edge_index_dict,
        metapath=metapath,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=2*(len(metapath) + 1) + 1,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True,
    )

    return model