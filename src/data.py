# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

class CharDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_path = cfg.data.data_path
        self.block_size = cfg.data.block_size
        self.batch_size = cfg.data.batch_size
        
        # Sẽ được khởi tạo trong setup()
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}

    def setup(self, stage=None):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [self.stoi[c] for c in s]
        data = torch.tensor(encode(text), dtype=torch.long)
        
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def train_dataloader(self):
        train_dataset = CharDataset(self.train_data, self.block_size)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        val_dataset = CharDataset(self.val_data, self.block_size)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == '__main__':
    print("hello")