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
                 dtype='bfloat16', gradient_checkpointing=True, generation_use_cache=False):
        nn.Module.__init__(self)
        dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                  'float16': torch.float16}
        if dtype not in dtypes:
            raise ValueError(f'Unsupported dtype {dtype!r}; choose {sorted(dtypes)}')
        if type(generation_use_cache) is not bool:
            raise ValueError('generation_use_cache must be a boolean')
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir() or not (path / 'config.json').is_file():
            raise ValueError('model_path must be a complete local HF model directory; downloads are disabled')
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        # An inference execution option, not a parameter/checkpoint schema change.
        # False preserves historical full-prefix generation for old configs.
        self.generation_use_cache = generation_use_cache
        if config.risk_horizon_schema == '0p5_1_2_v2':
            self.register_buffer('_risk_horizon_contract',torch.tensor([.5,1.,2.,2.],dtype=torch.float64))
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
        self._initialize_action_head(self.llm_hidden_dim)

    def _native_model(self):
        return self.language_model.get_base_model() if self.use_lora else self.language_model

    def _action_query_state(self, prefix, padding, ego_token):
        hidden = torch.cat((prefix, ego_token[:, None].to(prefix.dtype),
                            self.action_query[None, None].expand(len(prefix), 1, -1).to(prefix.dtype)), 1)
        if hidden.shape[1] > self.config.max_length:
            raise ValueError('action query exceeds configured context length')
        padding = torch.cat((padding, padding.new_zeros(len(prefix), 2)), 1)
        attention = (~padding).long()
        positions = (attention.cumsum(-1)-1).clamp_min(0)
        result = self._native_model().base_model(inputs_embeds=hidden, attention_mask=attention,
            position_ids=positions, use_cache=False, return_dict=True)
        return result.last_hidden_state[:, -1]

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
        risk = self.condition_risk(risk)
        if any(len(text.encode('utf-8')) > self.config.max_instruction_bytes for text in instructions):
            raise ValueError('Instruction exceeds max_instruction_bytes; increase the explicit context budget')
        rows = [self.native_tokenizer.encode(text, add_special_tokens=False) for text in instructions]
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
        parts = [self.radar_projector(radar_tokens.to(adapter_dtype)).to(dtype)]
        if self.config.use_risk_token:
            parts.append(self.risk_projector((risk / scale).to(adapter_dtype)).unsqueeze(1).to(dtype))
        parts.append(self._embed_native(ids))
        prefix = torch.cat(parts, dim=1)
        mask = torch.cat((torch.zeros((b, radar_tokens.shape[1] + int(self.config.use_risk_token)), dtype=torch.bool,
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
        # HF gates checkpointing on the checkpoint-owning module's training
        # flag. RL must keep attention and LoRA dropout disabled while scoring
        # with gradients. Set only those gate flags directly; .train() would
        # recursively turn stochastic child layers back on. Non-reentrant
        # recomputation invokes the unchanged eval-mode decoder children.
        checkpoint_gates = []
        if torch.is_grad_enabled() and not self.training:
            checkpoint_gates = [module for module in backbone.modules()
                                if getattr(module, 'gradient_checkpointing', False)
                                and not module.training]
            for module in checkpoint_gates:
                module.training = True
        try:
            result = backbone(inputs_embeds=hidden, attention_mask=attention_mask,
                              position_ids=positions, use_cache=False, return_dict=True)
        finally:
            for module in checkpoint_gates:
                module.training = False
        selected = result.last_hidden_state[:, prefix.shape[1]:]
        return self.physical_head(selected.to(self.physical_head.weight.dtype)).float()

    def _generation_decoder(self, prefix, prefix_mask):
        """Use a fresh KV cache for this observation, shared by all generation modes.

        Prefill contains continuous conditions, padded text and physical BOS.
        Later calls append one physical embedding. RoPE positions count valid
        tokens; cache positions count actual slots, including masked text pads.
        Training and policy rescoring continue to call the full `_decode_inputs`.
        """
        if not self.generation_use_cache:
            return super()._generation_decoder(prefix, prefix_mask)
        if self.training:
            raise ValueError('Cached generation requires eval mode; training/rescoring use the full path')
        backbone = self._native_model().base_model
        past = None
        consumed = 0

        @torch.no_grad()
        def decode(input_ids):
            nonlocal past, consumed
            if input_ids.ndim != 2 or input_ids.shape != (prefix.shape[0], consumed + 1):
                raise ValueError('Cached generation must append exactly one physical token per step')
            length = prefix.shape[1] + input_ids.shape[1]
            if length > self.config.max_length:
                raise ValueError(f'sequence length {length} exceeds max_length={self.config.max_length}')
            new_ids = input_ids[:, -1:]
            if ((new_ids < 0) | (new_ids >= self.tokenizer.vocab_size)).any():
                raise ValueError('input_ids outside the physical vocabulary')
            hidden = self.physical_embedding(new_ids).to(prefix.dtype)
            if consumed == 0:
                hidden = torch.cat((prefix, hidden), dim=1)
            padding = torch.cat((prefix_mask, input_ids == self.tokenizer.pad_id), dim=1)
            attention_mask = (~padding).long()
            positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)[:, -hidden.shape[1]:]
            cache_positions = torch.arange(length - hidden.shape[1], length, device=hidden.device)
            result = backbone(inputs_embeds=hidden, attention_mask=attention_mask,
                              position_ids=positions, cache_position=cache_positions,
                              past_key_values=past, use_cache=True, return_dict=True)
            past = result.past_key_values
            if past is None:
                raise RuntimeError('HF backbone did not return the requested generation cache')
            consumed += 1
            return self.physical_head(result.last_hidden_state[:, -1:].to(self.physical_head.weight.dtype)).float()

        return decode

    def checkpoint_state(self):
        """Store only trainable adapters; reload the frozen base from model_path."""
        return dict(format='radar_vla_hf_adapters_v2', lora=self.use_lora,
                    risk_horizon_schema=self.config.risk_horizon_schema,
                    action_representation=self.config.action_representation,
                    action_contract=(self._action_representation_contract.detach().cpu().clone()
                                     if hasattr(self, '_action_representation_contract') else None),
                    llm_hidden_dim=self.llm_hidden_dim,
                    physical_hf_ids=self.physical_hf_ids.detach().cpu().clone(),
                    parameters={name: parameter.detach().cpu().clone()
                                for name, parameter in self.named_parameters() if parameter.requires_grad})

    def load_checkpoint_state(self, state):
        if state.get('action_representation', 'token_spline_v1') != self.config.action_representation:
            raise ValueError('HF checkpoint action representation incompatible')
        incoming = state.get('action_contract')
        expected_action = getattr(self, '_action_representation_contract', None)
        if ((incoming is None) != (expected_action is None)
                or incoming is not None and not torch.equal(incoming.cpu(), expected_action.cpu())):
            raise ValueError('HF checkpoint action representation contract incompatible')
        if state.get('risk_horizon_schema','1_2_3_v1') != self.config.risk_horizon_schema:
            raise ValueError('HF checkpoint Risk horizons incompatible')
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if (state.get('format') != 'radar_vla_hf_adapters_v2'
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
