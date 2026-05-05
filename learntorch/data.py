import learntorch as lt
import numpy as np

class DataLoader:

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size):
        self.shape_x = list(x.shape)
        self.shape_y = list(y.shape)
        self.x = lt.Tensor(list(x.shape), x.flatten().tolist())
        self.y = lt.Tensor(list(y.shape), y.flatten().tolist())
        self.batch_size = batch_size
        self._n = x.shape[0]

    def __len__(self):
        return (self._n + self.batch_size - 1 ) // self.batch_size

    def __iter__(self):
        for i in range(0, self._n, self.batch_size):
            end: int = min(i + self.batch_size, self._n)
            yield self.x.slice(i, end), self.y.slice(i, end)
        
def fit(trainer: lt.Trainer, train_loader: DataLoader, val_loader: DataLoader, 
        epochs: int, print_every=10):
    trainer.fit(train_loader.x,train_loader.y, val_loader.x, val_loader.y,
                epochs, train_loader.batch_size, print_every=print_every)