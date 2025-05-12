import torch
import torch_geometric as pyg
from torch_geometric.nn import ComplEx, DistMult, KGEModel, TransE
from tqdm import tqdm

from src.utils import EarlyStopping


def train_kge(
        model: KGEModel,
        loader: pyg.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        log_steps: int = 100, 
        max_epochs: int = 5,
        patience: int = 5,
        early_stopping_thr: float = 1e-4,
        verbose: bool = False) -> list[float]:

    best_state = None

    train_losses = []
    early_stopper = EarlyStopping(patience=patience, threshold=early_stopping_thr)
    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = total_examples = 0
        for i, (head_index, rel_type, tail_index) in (pbar := tqdm(enumerate(loader), total=len(loader), desc=f"Epoch: {epoch}", disable=not verbose)):
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * head_index.numel()
            total_examples += head_index.numel()

            if (i + 1) % log_steps == 0:
                pbar.set_postfix(step=f'{i + 1:05d}/{len(loader)}',
                    loss=f'{total_loss / total_examples:.4f}')


        avg_loss = total_loss / total_examples
        
        train_losses.append(avg_loss)
        
        scheduler.step(avg_loss)

        # Early-stopping
        early_stopper.step(model, avg_loss)
        if early_stopper.should_stop():
            pbar.set_postfix(step=f'{i + 1:05d}/{len(loader)}',
                             loss=f'{avg_loss:.4f}',
                             early_stopped='True')
            break

    if early_stopper.best_state is not None:
        with torch.no_grad():
            model.load_state_dict(early_stopper.best_state)

    return train_losses


def setup_kge(
        data: pyg.data.Data,
        embedding_dim: int = 64,
        model_type: KGEModel = TransE) -> KGEModel:

    kge_model = model_type(
        num_nodes=data.num_nodes,
        num_relations=data.num_edge_types,
        hidden_channels=embedding_dim
    )
    return kge_model


def get_kge_models() -> dict:
    return dict(zip(['transE', 'distmult', 'complEx'], [TransE, DistMult, ComplEx]))