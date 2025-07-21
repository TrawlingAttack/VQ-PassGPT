import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, beta=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, z):
        z_flattened = z.view(-1, self.code_dim)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        commitment_loss = F.mse_loss(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        vq_loss = self.beta * commitment_loss + codebook_loss

        z_q = z + (z_q - z).detach()
        return z_q, vq_loss

class VQPassGPTModel(GPT2LMHeadModel):
    def __init__(self, config, use_vq=True):
        super().__init__(config)
        #self.transformer = GPT2Model(config) 
        self.use_vq = use_vq

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        self.dropout = nn.Dropout(config.embd_pdrop)

        if self.use_vq:
            self.vq1 = VectorQuantizer(num_codes=512, code_dim=config.n_embd)
            self.vq2 = VectorQuantizer(num_codes=512, code_dim=config.n_embd)

        self.decoder = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # luôn lấy dict để dễ xử lý
        )

        hidden_states = transformer_outputs.last_hidden_state

        vq_loss1 = vq_loss2 = 0.0
        if self.use_vq:
            hidden_states, vq_loss1 = self.vq1(hidden_states)
            hidden_states, vq_loss2 = self.vq2(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = lm_loss + vq_loss1 + vq_loss2 if self.use_vq else lm_loss

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def num_parameters(self, exclude_embeddings: bool = False):
        if exclude_embeddings:
            excluded = {id(p) for p in self.embed_tokens.parameters()}
            excluded.update(id(p) for p in self.lm_head.parameters())
            return sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in excluded)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
