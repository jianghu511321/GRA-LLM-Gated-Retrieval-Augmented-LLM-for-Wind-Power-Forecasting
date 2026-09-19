import torch
from torch.utils.data import Dataset
import faiss

class GatedRAFDataset(Dataset):
    def __init__(self, x_data, y_data, kb_x, kb_y, k=3):
        self.x_data = x_data
        self.y_data = y_data
        self.kb_x = kb_x 
        self.kb_y = kb_y
        self.k = k
        
        print(f"Building FAISS Index (K={k})...")
        d = x_data.shape[1] * x_data.shape[2] 
        
        # Flatten Knowledge Base sequences for FAISS L2 search
        if isinstance(kb_x, torch.Tensor): kb_flat_idx = kb_x.cpu().numpy()
        else: kb_flat_idx = kb_x
        kb_flat_idx = kb_flat_idx.reshape(len(kb_flat_idx), -1).astype('float32')

        # Flatten Query sequences
        if isinstance(x_data, torch.Tensor): query_flat_idx = x_data.cpu().numpy()
        else: query_flat_idx = x_data
        query_flat_idx = query_flat_idx.reshape(len(query_flat_idx), -1).astype('float32')
        
        # Initialize and populate FAISS L2 Index
        index = faiss.IndexFlatL2(d)
        index.add(kb_flat_idx)
        
        # Perform retrieval
        D, I = index.search(query_flat_idx, k)
        self.match_indices = I

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        query_seq = self.x_data[idx] 
        neighbor_indices = self.match_indices[idx]
        
        k_seqs = self.kb_x[neighbor_indices] 
        k_labels = self.kb_y[neighbor_indices]
        target = self.y_data[idx]
        
        return {
            'query_x': torch.tensor(query_seq, dtype=torch.float32),
            'retrieved_x': torch.tensor(k_seqs, dtype=torch.float32),
            'retrieved_y': torch.tensor(k_labels, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32).view(-1)
        }