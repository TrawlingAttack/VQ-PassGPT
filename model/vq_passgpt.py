import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class VQPassGPTModel(GPT2LMHeadModel):
    """
    - Decoder-only GPT-2 + 1 bottleneck VQ cho phần pattern.
    - VQ codebook = các vector embedding của TẬP TOKEN HỢP LỆ CHO PATTERN (ID 5..40),
      tức là nearest-neighbor vào wte[5..40] (không học codebook riêng).
    - Password không qua VQ.
    - (Tuỳ chọn) vẫn có mask cấu trúc trong train để loss chỉ "thấy" tập hợp hợp lệ từng pha.
    """

    def __init__(self, config, use_vq: bool = True):
        super().__init__(config)
        self.use_vq = bool(use_vq)

        # Đọc id đặc biệt từ config (fallback an toàn)
        def _norm_id(x, default):
            return int(x) if x is not None else int(default)

        self.bos_token_id = _norm_id(getattr(config, "bos_token_id", None), 0)
        self.sep_token_id = _norm_id(getattr(config, "sep_token_id", None), 1)
        self.eos_token_id = _norm_id(getattr(config, "eos_token_id", None), 2)
        self.pad_token_id = _norm_id(getattr(config, "pad_token_id", None), 4)

        # Miền ID (đúng theo vocab của bạn)
        self.pattern_ids  = torch.arange(5, 41)    # [5..40]
        self.password_ids = torch.arange(41, 135)  # [41..134]

        # Bật/tắt mask cấu trúc trong train
        self.enforce_constraints_train = bool(getattr(config, "enforce_constraints_train", True))

        # Hệ số VQ (commitment loss)
        self.vq_beta = float(getattr(config, "vq_beta", 0.1))

        # Kiểm tra nhanh
        if self.config.vocab_size is not None:
            vmax = int(self.config.vocab_size) - 1
            assert int(self.pattern_ids.max()) <= vmax and int(self.password_ids.max()) <= vmax, \
                "pattern/password IDs vượt quá vocab_size!"

    # ------ VQ trên không gian embedding của token pattern ------
    @torch.no_grad()
    def _nearest_pattern_embed(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (L, D) vector ẩn của các vị trí pattern (một sample)
        Trả về z_q: (L, D) là các vector embedding gần nhất trong wte[pattern_ids].
        """
        # Lấy ma trận embedding ứng với các token pattern (K, D)
        codebook = self.transformer.wte.weight[self.pattern_ids.to(z.device)]  # không học codebook riêng

        # Khoảng cách bình phương: ||z||^2 - 2 z E^T + ||E||^2
        zf = z.view(-1, z.size(-1))                         # (L, D)
        zz = (zf ** 2).sum(dim=1, keepdim=True)             # (L, 1)
        ee = (codebook ** 2).sum(dim=1).unsqueeze(0)        # (1, K)
        distances = zz - 2.0 * (zf @ codebook.t()) + ee     # (L, K)
        idx = torch.argmin(distances, dim=1)                # (L,)

        z_q = codebook.index_select(0, idx).view_as(z)      # (L, D)
        return z_q

    def _apply_structural_mask(self, shift_logits, shift_inputs):
        """
        Trước <SEP>: chỉ cho phép pattern_ids ∪ {SEP}
        Sau  <SEP>: chỉ cho phép password_ids ∪ {EOS}
        """
        B, Tm1, V = shift_logits.size()
        device = shift_logits.device

        allowed_before = torch.full((V,), float("-inf"), device=device)
        allowed_after  = torch.full((V,), float("-inf"), device=device)

        idx_before = torch.cat([self.pattern_ids.to(device), torch.tensor([self.sep_token_id], device=device)])
        idx_after  = torch.cat([self.password_ids.to(device), torch.tensor([self.eos_token_id], device=device)])
        allowed_before.index_fill_(0, idx_before, 0.0)
        allowed_after.index_fill_(0, idx_after, 0.0)

        for b in range(B):
            seen_sep = False
            for t in range(Tm1):
                shift_logits[b, t] = shift_logits[b, t] + (allowed_after if seen_sep else allowed_before)
                if int(shift_inputs[b, t].item()) == self.sep_token_id:
                    seen_sep = True
        return shift_logits

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

        # Backbone GPT-2
        outputs = self.transformer(
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
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, D)

        # -------- VQ cho phần pattern --------
        vq_loss = hidden_states.new_zeros(())
        if self.use_vq and input_ids is not None:
            B, T, D = hidden_states.size()
            new_hidden = hidden_states.clone()
            for b in range(B):
                # vị trí <SEP> đầu tiên
                pos = (input_ids[b] == self.sep_token_id).nonzero(as_tuple=False)
                if len(pos) == 0:
                    continue
                sep_idx = int(pos[0])
                # Lấy đoạn [0..sep_idx] (bao gồm BOS..SEP) để VQ về codebook pattern
                z = hidden_states[b, :sep_idx + 1, :]               # (L, D)
                with torch.no_grad():
                    z_q = self._nearest_pattern_embed(z)            # lượng tử hoá NN về wte[5..40]
                # straight-through trick: (z_q - z).detach() + z
                new_hidden[b, :sep_idx + 1, :] = z + (z_q - z).detach()
                # commitment loss (codebook không học → không cần codebook loss)
                vq_loss = vq_loss + self.vq_beta * F.mse_loss(z, z_q.detach())
            hidden_states = new_hidden

        # Head logits
        logits = self.lm_head(hidden_states)

        # -------- Loss --------
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Mask cấu trúc trong train để loss chỉ tính trên tập hợp hợp lệ
            if self.training and self.enforce_constraints_train:
                shift_inputs = (input_ids if input_ids is not None else labels)[..., :-1].contiguous()
                shift_logits = self._apply_structural_mask(shift_logits, shift_inputs)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
            loss = lm_loss + (vq_loss if self.use_vq else 0.0)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ----- generation hooks -----
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    # FLOPs estimate helper
    def num_parameters(self, exclude_embeddings: bool = False):
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        excluded = {id(p) for p in self.transformer.wte.parameters()}
        excluded |= {id(p) for p in self.lm_head.parameters()}
        return sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in excluded)
