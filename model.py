import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from config import *

class AttentionRAF_GPT2(nn.Module):
    def __init__(self, original_dim=3, k=3, output_dim=1):
        super(AttentionRAF_GPT2, self).__init__()
        
        print(f"Initializing Fine-Grained GRA-LLM Architecture...")
        self.k = k
        d_model = 128  
        self.embed_dim = d_model  
        self.seq_len = SEQ_LEN
        
        # Lightweight GPT-2 configuration for time-series
        self.config = GPT2Config(
            n_embd=d_model,       
            n_layer=4,            
            n_head=4,             
            n_positions=1024,
            vocab_size=1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        
        self.gpt2 = GPT2Model(self.config)

        self.input_projector = nn.Linear(original_dim, self.embed_dim)
        self.label_projector = nn.Linear(1, self.embed_dim)
        
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim)
        
        gate_input_dim = self.seq_len * original_dim * 2
        
        # Dynamic Trust Gate
        self.trust_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.head = nn.Linear(self.embed_dim, output_dim)

        # Explicitly initialize the bias of the final linear layer to a large negative scalar.
        # This forces the gate to output values near 0 initially.
        with torch.no_grad():
            self.trust_gate[-2].bias.fill_(-5.0) 
        
    def forward(self, query_x, retrieved_x, retrieved_y):
        batch_size, seq_len, _ = query_x.shape
        
        # A. Feature Embedding
        query_emb = self.input_projector(query_x) 
        kb_flat = retrieved_x.view(-1, seq_len, query_x.shape[-1])
        kb_emb = self.input_projector(kb_flat).view(batch_size, self.k, seq_len, self.embed_dim)

        # B. Scaled Dot-Product Attention Interaction
        q_proj = self.W_q(query_emb).unsqueeze(1) 
        k_proj = self.W_k(kb_emb)                 
        
        interaction = q_proj * k_proj 
        raw_scores = interaction.sum(dim=[2, 3]).unsqueeze(1) 
        scale_factor = (self.embed_dim * seq_len) ** 0.5 
        attn_weights = F.softmax(raw_scores / scale_factor, dim=-1)

        # C. Attention-Weighted External Context
        w_expanded_x = attn_weights.view(batch_size, self.k, 1, 1)
        weighted_kb_emb = (w_expanded_x * kb_emb).sum(dim=1) 
        
        w_expanded_y = attn_weights.view(batch_size, self.k)
        weighted_history_val = (w_expanded_y * retrieved_y).sum(dim=1) 

        # D. Dynamic Trust Gate Calculation
        q_flat = query_x.view(batch_size, -1) 
        w_expanded_raw = attn_weights.view(batch_size, self.k, 1, 1)
        weighted_kb_raw = (w_expanded_raw * retrieved_x).sum(dim=1) 
        kb_flat = weighted_kb_raw.view(batch_size, -1) 
        
        gate_input = torch.cat([q_flat, kb_flat], dim=-1) 
        gate_val = self.trust_gate(gate_input) 
        gate_val_seq = gate_val.unsqueeze(1) 

        # E. Fusion & Autoregressive Inference
        fused_input = query_emb + gate_val_seq * weighted_kb_emb 
        outputs = self.gpt2(inputs_embeds=fused_input)
        
        # Extract the hidden state of the last time step
        last_hidden = outputs.last_hidden_state[:, -1, :] 
        
        y_embedding = self.label_projector(weighted_history_val.unsqueeze(1)) 
        final_feature = last_hidden + gate_val * y_embedding
        
        final_prediction = self.head(final_feature)
        
        return final_prediction