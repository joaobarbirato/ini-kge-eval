import pandas as pd


from datetime import datetime
from os import path as os_path


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
