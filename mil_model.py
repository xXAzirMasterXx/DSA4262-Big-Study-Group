import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
UNKNOWN_KMER = 1024


def encode_5mer(kmer: str) -> int:
    if len(kmer) != 5 or any(base not in BASE_TO_INT for base in kmer):
        return UNKNOWN_KMER
    value = 0
    for base in kmer:
        value = value * 4 + BASE_TO_INT[base]
    return value


class SiteBagDataset(Dataset):
    def __init__(self, sites, read_maps, n_reads_per_site, feature_mean, feature_std):
        self.sites = sites.reset_index(drop=True)
        self.read_maps = read_maps
        self.n_reads_per_site = int(n_reads_per_site)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(
            feature_std, dtype=np.float32), 1e-6)

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, index):
        row = self.sites.iloc[index]
        dataset_number = int(row["dataset"])
        read_start = int(row["read_start"])
        available = int(row["n_reads"])
        reads = self.read_maps[dataset_number][read_start: read_start + available]
        replace = available < self.n_reads_per_site
        selected = np.random.choice(
            available,
            size=self.n_reads_per_site,
            replace=replace,
        )
        sampled_reads = np.asarray(reads[selected], dtype=np.float32)
        sampled_reads = (sampled_reads - self.feature_mean) / self.feature_std

        sequence = str(row["sequence"]).upper()
        kmers = [sequence[offset: offset + 5] for offset in range(3)]
        kmer_ids = np.asarray([encode_5mer(kmer)
                              for kmer in kmers], dtype=np.int64)

        return (
            int(index),
            torch.from_numpy(np.ascontiguousarray(sampled_reads)),
            torch.from_numpy(kmer_ids),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )


class M6ANetInspiredMIL(nn.Module):
    def __init__(self, embed_dim=2, hidden_dim=150, dropout=0.1):
        super().__init__()
        self.kmer_embedding = nn.Embedding(1025, embed_dim)
        input_dim = 9 + 3 * embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        )
        self.read_probability = nn.Linear(32, 1)

    def read_probabilities(self, read_features, kmer_ids):
        batch_size, n_reads, _ = read_features.shape
        kmer_embeddings = self.kmer_embedding(kmer_ids).reshape(batch_size, -1)
        kmer_embeddings = kmer_embeddings.unsqueeze(1).expand(-1, n_reads, -1)
        read_inputs = torch.cat([read_features, kmer_embeddings], dim=-1)
        read_inputs = read_inputs.reshape(batch_size * n_reads, -1)
        hidden = self.encoder(read_inputs)
        read_probabilities = torch.sigmoid(self.read_probability(hidden))
        return read_probabilities.reshape(batch_size, n_reads)

    def forward(self, read_features, kmer_ids):
        return noisy_or(self.read_probabilities(read_features, kmer_ids))


def noisy_or(read_probabilities):
    """Stable probability that at least one read in a bag is modified."""
    return -torch.expm1(
        torch.log1p(-read_probabilities.clamp(max=1.0 - 1e-7)).sum(dim=-1)
    )
