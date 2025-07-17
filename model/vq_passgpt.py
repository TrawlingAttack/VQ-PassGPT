# model/vq_passgpt.py

import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Config, GPT2Model
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, z):
        # z: (B, T, D)
        z_flattened = z.view(-1, self.code_dim)  # (B*T, D)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )  # (B*T, K)

        encoding_indices = torch.argmin(distances, dim=1)  # (B*T)
        z_q = self.embedding(encoding_indices).view(z.shape)  # (B, T, D)

        # Compute VQ loss
        commitment_loss = F.mse_loss(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        vq_loss = self.beta * commitment_loss + codebook_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss

class VQPassGPTModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        self.dropout = nn.Dropout(config.embd_pdrop)

        # Vector Quantizers
        self.vq1 = VectorQuantizer(num_codes=512, code_dim=config.n_embd)
        self.vq2 = VectorQuantizer(num_codes=512, code_dim=config.n_embd)

        # GPT2 Decoder
        self.decoder = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        batch_size, seq_length = input_ids.size()

        pos_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.embed_tokens(input_ids)  # (B, T, D)
        x = token_embeds + self.pos_embed[:, :seq_length, :]
        x = self.dropout(x)

        # Apply first quantization layer
        x, vq_loss1 = self.vq1(x)

        # Pass through first half of decoder
        # Cast attention mask only once
        if attention_mask is not None:
        # Ensure attention_mask is float32 and shaped [B, 1, 1, T]
            attention_mask = attention_mask.to(dtype=x.dtype)
            attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)


        # First half of decoder
        half_layers = self.config.n_layer // 2
        for i, block in enumerate(self.decoder.h[:half_layers]):
            x = block(x, attention_mask=attention_mask)[0]



        # Apply second quantization layer
        x, vq_loss2 = self.vq2(x)

        # Pass through remaining decoder layers
        for i, block in enumerate(self.decoder.h[half_layers:]):
            x = block(x, attention_mask=attention_mask)[0]

        x = self.decoder.ln_f(x)  # Final layer norm
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = lm_loss + vq_loss1 + vq_loss2

        return {
            "loss": loss,
            "logits": logits,
            "vq_loss": vq_loss1 + vq_loss2
        }

    def num_parameters(self, exclude_embeddings: bool = False):
    # Nếu exclude_embeddings=True, loại bỏ embedding layer
        if exclude_embeddings:
        # Loại bỏ embed_tokens và lm_head nếu cần
            excluded = {id(p) for p in self.embed_tokens.parameters()}
            if hasattr(self, 'lm_head'):
                excluded.update(id(p) for p in self.lm_head.parameters())
            return sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in excluded)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

