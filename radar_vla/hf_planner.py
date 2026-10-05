"""Local pretrained causal LM + LoRA backend for the RadarVLA physical grammar.

The native pretrained transformer reads radar/risk embeddings and its own text
tokenizer. A small trainable physical input/output vocabulary avoids unfreezing
Qwen's full embedding table or materializing full-vocabulary logits. No model
download, remote Python code, or automatic device placement is permitted here.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .planner import PlannerConfig, RiskConditionedPlanner
from .tokenizer import PhysicalTokenizer


class HFRiskConditionedPlanner(RiskConditionedPlanner):
    """Compatible forward/generate/decode API backed by a local HF causal LM.

    ``lora=False`` freezes the native LM and trains only continuous-prefix and
    physical-vocabulary adapters; it does not silently enable full fine-tuning.
    ``config.hidden_dim`` belongs to the prototype interface. The pretrained
    hidden dimension comes exclusively from the local language-model config.
    """

    def __init__(self, config: PlannerConfig, model_path, lora=True,
                 dtype='bfloat16', gradient_checkpointing=True):
        nn.Module.__init__(self)
        dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                  'float16': torch.float16}
        if dtype not in dtypes:
            raise ValueError(f'Unsupported dtype {dtype!r}; choose {sorted(dtypes)}')
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir() or not (path / 'config.json').is_file():
            raise ValueError('model_path must be a complete local HF model directory; downloads are disabled')
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.model_path = str(path)
        self.use_lora = bool(lora)
        self.tokenizer = PhysicalTokenizer(config.bins, config.coordinate_limit_m, config.velocity_limit_mps)
        self.native_tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False)
        native = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            torch_dtype=dtypes[dtype], attn_implementation='sdpa')
        self.llm_hidden_dim = int(native.get_input_embeddings().weight.shape[1])
        names = [f'<|radar_vla_{index:04d}|>' for index in range(self.tokenizer.vocab_size)]
        self.native_tokenizer.add_special_tokens({'additional_special_tokens': names})
        native.resize_token_embeddings(len(self.native_tokenizer), mean_resizing=False)
        physical_ids = torch.tensor(self.native_tokenizer.convert_tokens_to_ids(names), dtype=torch.long)
        if len(set(physical_ids.tolist())) != self.tokenizer.vocab_size:
            raise ValueError('Native tokenizer did not register a unique physical vocabulary')
        self.register_buffer('physical_hf_ids', physical_ids)
        lookup = torch.full((len(self.native_tokenizer),), -1, dtype=torch.long)
        lookup[physical_ids] = torch.arange(self.tokenizer.vocab_size)
        self.register_buffer('native_to_physical', lookup, persistent=False)
        native.requires_grad_(False)
        native.config.use_cache = False
        if gradient_checkpointing:
            # Non-reentrant checkpointing supports trainable continuous prefixes
            # and prevents DDP's re-entrant "marked ready twice" failure.
            native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.physical_embedding = nn.Embedding.from_pretrained(
            native.get_input_embeddings().weight.detach()[physical_ids].clone(),
            freeze=False, padding_idx=self.tokenizer.pad_id)
        self.physical_head = nn.Linear(self.llm_hidden_dim, self.tokenizer.vocab_size,
                                       bias=False, dtype=dtypes[dtype])
        with torch.no_grad():
            self.physical_embedding.weight[self.tokenizer.pad_id].zero_()
            self.physical_head.weight.copy_(native.get_output_embeddings().weight.detach()[physical_ids])
        if lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError('LoRA requires peft in the independent RadarVLA environment') from exc
            candidates = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
            present = {name.rsplit('.', 1)[-1] for name, _ in native.named_modules()}
            targets = sorted(candidates & present)
            if not targets:
                raise ValueError('LoRA backend requires Qwen/Llama-style attention and MLP projections')
            native = get_peft_model(native, LoraConfig(task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=.05, target_modules=targets, bias='none'))
        self.language_model = native
        # Adapters remain float32 for optimizer stability; their outputs are cast
        # to the pretrained model dtype at the transformer boundary.
        self.radar_projector = nn.Linear(config.radar_dim, self.llm_hidden_dim)
        self.risk_projector = nn.Sequential(nn.Linear(5, self.llm_hidden_dim), nn.GELU(),
                                           nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim))
        self.physical_embedding.float()
        self.physical_head.float()

    def _native_model(self):
        return self.language_model.get_base_model() if self.use_lora else self.language_model

    def _embed_native(self, ids):
        embedded = self._native_model().get_input_embeddings()(ids)
        physical = self.native_to_physical[ids]
        # Namespaced physical tokens appearing in an instruction use the same
        # checkpointed trainable rows, not randomly initialized frozen HF rows.
        return torch.where((physical >= 0)[..., None],
                           self.physical_embedding(physical.clamp_min(0)).to(embedded.dtype), embedded)

    def encode_prefix(self, radar_tokens, risk, instructions):
        if radar_tokens.ndim != 3 or radar_tokens.shape[-1] != self.config.radar_dim:
            raise ValueError('radar_tokens must have shape [batch,tokens,radar_dim]')
        b = radar_tokens.shape[0]
        if risk.shape != (b, 5) or len(instructions) != b:
            raise ValueError('risk and instruction batch sizes must match radar_tokens')
        if not torch.isfinite(radar_tokens).all() or not torch.isfinite(risk).all():
            raise ValueError('radar and risk conditions must be finite')
        # Keep the existing config's byte-budget semantics; tokenize the retained
        # text with the native Qwen tokenizer rather than the prototype byte IDs.
        texts = [text.encode('utf-8')[:self.config.max_instruction_bytes].decode('utf-8', errors='ignore')
                 for text in instructions]
        rows = [self.native_tokenizer.encode(text, add_special_tokens=False) for text in texts]
        width = max(1, max(map(len, rows)))
        pad = self.native_tokenizer.pad_token_id
        if pad is None:
            pad = self.native_tokenizer.eos_token_id
        if pad is None:
            pad = 0
        ids = torch.full((b, width), pad, dtype=torch.long, device=radar_tokens.device)
        text_mask = torch.ones((b, width), dtype=torch.bool, device=radar_tokens.device)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=ids.device)
            text_mask[i, :len(row)] = False
        dtype = self._native_model().get_input_embeddings().weight.dtype
        adapter_dtype = self.radar_projector.weight.dtype
        scale = risk.new_tensor([1., 1., 1., self.config.coordinate_limit_m, 10.])
        prefix = torch.cat((self.radar_projector(radar_tokens.to(adapter_dtype)).to(dtype),
            self.risk_projector((risk / scale).to(adapter_dtype)).unsqueeze(1).to(dtype),
            self._embed_native(ids)), dim=1)
        mask = torch.cat((torch.zeros((b, radar_tokens.shape[1] + 1), dtype=torch.bool,
                                       device=ids.device), text_mask), dim=1)
        return prefix, mask

    def _decode_inputs(self, prefix, prefix_mask, input_ids):
        length = prefix.shape[1] + input_ids.shape[1]
        if length > self.config.max_length:
            raise ValueError(f'sequence length {length} exceeds max_length={self.config.max_length}')
        if ((input_ids < 0) | (input_ids >= self.tokenizer.vocab_size)).any():
            raise ValueError('input_ids outside the physical vocabulary')
        hidden = torch.cat((prefix, self.physical_embedding(input_ids).to(prefix.dtype)), dim=1)
        padding = torch.cat((prefix_mask, input_ids == self.tokenizer.pad_id), dim=1)
        attention_mask = (~padding).long()
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        # The LoRA-injected pretrained body returns only hidden states. Calling
        # the full causal-LM head would allocate [B,L,~150k] unnecessary logits.
        backbone = self._native_model().base_model
        result = backbone(inputs_embeds=hidden, attention_mask=attention_mask,
                          position_ids=positions, use_cache=False, return_dict=True)
        selected = result.last_hidden_state[:, prefix.shape[1]:]
        return self.physical_head(selected.to(self.physical_head.weight.dtype)).float()

    def checkpoint_state(self):
        """Store only trainable adapters; reload the frozen base from model_path."""
        return dict(format='radar_vla_hf_adapters_v1', lora=self.use_lora,
                    llm_hidden_dim=self.llm_hidden_dim,
                    physical_hf_ids=self.physical_hf_ids.detach().cpu().clone(),
                    parameters={name: parameter.detach().cpu().clone()
                                for name, parameter in self.named_parameters() if parameter.requires_grad})

    def load_checkpoint_state(self, state):
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if (state.get('format') != 'radar_vla_hf_adapters_v1'
                or state.get('lora') != self.use_lora
                or state.get('llm_hidden_dim') != self.llm_hidden_dim):
            raise ValueError('HF adapter checkpoint configuration differs')
        if not torch.equal(state['physical_hf_ids'].cpu(), self.physical_hf_ids.cpu()):
            raise ValueError('HF physical-token mapping differs from checkpoint')
        if set(state['parameters']) != set(expected):
            raise ValueError('HF checkpoint trainable parameter keys differ')
        for name, parameter in expected.items():
            value = state['parameters'][name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint tensor {name}')
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state['parameters'][name].to(parameter))
