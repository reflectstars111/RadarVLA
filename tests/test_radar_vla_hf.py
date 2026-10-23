"""Real local Hugging Face model integration; no downloads or mocked backbone."""

import pytest
import torch


@pytest.fixture
def tiny_local_qwen(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    path = tmp_path / 'tiny_qwen'
    vocab = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
             'Drive': 4, 'safely': 5, 'turn': 6, 'left': 7}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
                                unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
    hf.save_pretrained(path)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0,
        bos_token_id=2, eos_token_id=3))
    model.save_pretrained(path)
    return path


def make_planner(path, **kwargs):
    from radar_vla.hf_planner import HFRiskConditionedPlanner
    from radar_vla.planner import PlannerConfig
    cfg = PlannerConfig(hidden_dim=8, radar_dim=8, num_heads=2, bins=8,
                        max_agents=1, horizon_steps=2, short_horizon_steps=1,
                        max_instruction_bytes=96, max_length=128)
    return HFRiskConditionedPlanner(cfg, path, dtype=kwargs.pop('dtype', 'float32'), **kwargs)


def observations():
    radar = torch.randn(2, 3, 8, requires_grad=True)
    risk = torch.tensor([[.1, .2, .3, 20., 1.], [.2, .3, .5, 5., 3.]], requires_grad=True)
    return radar, risk, ['Drive safely', 'turn left']


def test_local_hf_forward_uses_native_backbone_and_trains_prefix(tiny_local_qwen):
    from radar_vla.tokenizer import token_cross_entropy
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False)
    assert planner.llm_hidden_dim == 16
    assert planner.config.radar_dim == 8
    assert not any(p.requires_grad for p in planner.language_model.parameters())
    radar, risk, instructions = observations()
    targets = torch.tensor([[planner.tokenizer.token('<SHORT>'), planner.tokenizer.eos_id]] * 2)
    logits = planner(radar, risk, instructions, targets)
    assert logits.shape == (2, 2, planner.tokenizer.vocab_size)
    token_cross_entropy(logits, targets, planner.tokenizer).backward()
    assert torch.isfinite(radar.grad).all() and radar.grad.abs().sum() > 0
    assert torch.isfinite(risk.grad).all() and risk.grad.abs().sum() > 0
    assert planner.radar_projector.weight.grad.abs().sum() > 0
    assert planner.physical_embedding.weight.grad.abs().sum() > 0


def test_hf_generation_uses_same_physical_grammar(tiny_local_qwen):
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False).eval()
    radar, risk, instructions = observations()
    rows = planner.generate(radar, risk, instructions, max_new_tokens=80)
    assert len(rows) == 2
    for ids in rows:
        decoded = planner.decode(ids, future_times_s=[.5, 1.])
        assert decoded['valid']
        assert decoded['mode'] in ('short', 'long')


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_hf_response_policy_sampling_kl_and_lora_gradients(tiny_local_qwen, dtype):
    import copy
    from radar_vla.rl_policy import sample_plan, policy_statistics, validate_policy_rescore
    torch.manual_seed(12)
    planner = make_planner(tiny_local_qwen, lora=True, dtype=dtype, gradient_checkpointing=True).eval()
    reference = copy.deepcopy(planner).requires_grad_(False)
    radar, risk, instructions = observations()
    radar, risk, instructions = radar[:1].detach(), risk[:1].detach(), instructions[:1]
    sampled = sample_plan(planner, radar, risk, instructions, max_new_tokens=100)
    assert planner.decode(sampled['ids'], future_times_s=[.5, 1.])['valid']
    stats = policy_statistics(planner, radar, risk, instructions, sampled['ids'], reference=reference)
    validate_policy_rescore(planner, sampled, stats)
    assert stats['log_prob_sum'].item() == pytest.approx(sampled['log_prob_sum'], abs=5e-3)
    assert stats['kl'].item() == pytest.approx(0., abs=1e-6)
    (-stats['log_prob_sum'] + .02 * stats['kl']).backward()
    lora = [p for n, p in planner.named_parameters() if 'lora_' in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
    assert all(p.grad is None for p in reference.parameters())


def test_hf_lora_checkpoint_is_small_and_roundtrips(tiny_local_qwen):
    pytest.importorskip('peft')
    from radar_vla.tokenizer import token_cross_entropy
    planner = make_planner(tiny_local_qwen, lora=True, gradient_checkpointing=True)
    radar, risk, instructions = observations()
    targets = torch.tensor([[planner.tokenizer.token('<SHORT>'), planner.tokenizer.eos_id]] * 2)
    loss = token_cross_entropy(planner(radar, risk, instructions, targets), targets, planner.tokenizer)
    loss.backward()
    lora = [(n, p) for n, p in planner.named_parameters() if 'lora_' in n]
    assert lora and all(p.requires_grad for _, p in lora)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora)
    state = planner.checkpoint_state()
    keys = set(state['parameters'])
    assert keys == {n for n, p in planner.named_parameters() if p.requires_grad}
    assert not any('embed_tokens' in key or 'base_layer' in key for key in keys)
    restored = make_planner(tiny_local_qwen, lora=True, gradient_checkpointing=True)
    restored.load_checkpoint_state(state)
    planner.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(planner(radar, risk, instructions, targets),
                                   restored(radar, risk, instructions, targets), atol=0, rtol=0)
    broken = dict(state, parameters=dict(state['parameters']))
    broken['parameters'].pop(next(iter(keys)))
    with pytest.raises(ValueError, match='keys'):
        restored.load_checkpoint_state(broken)


def test_hf_requires_local_directory_and_explicit_supported_dtype(tmp_path):
    from radar_vla.hf_planner import HFRiskConditionedPlanner
    from radar_vla.planner import PlannerConfig
    with pytest.raises(ValueError, match='local'):
        HFRiskConditionedPlanner(PlannerConfig(), tmp_path / 'missing')
    with pytest.raises(ValueError, match='dtype'):
        HFRiskConditionedPlanner(PlannerConfig(), tmp_path, dtype='int8')


def test_bfloat16_backbone_accepts_float32_radar_and_preserves_adapter_gradients(tiny_local_qwen):
    pytest.importorskip('peft')
    from radar_vla.tokenizer import token_cross_entropy
    planner = make_planner(tiny_local_qwen, lora=True, dtype='bfloat16', gradient_checkpointing=True)
    assert planner._native_model().get_input_embeddings().weight.dtype == torch.bfloat16
    assert planner.radar_projector.weight.dtype == torch.float32
    radar, risk, instructions = observations()
    targets = torch.tensor([[planner.tokenizer.token('<LONG>'), planner.tokenizer.eos_id]] * 2)
    loss = token_cross_entropy(planner(radar, risk, instructions, targets), targets, planner.tokenizer)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(radar.grad).all() and radar.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in planner.parameters())


def test_local_hf_two_stage_pipeline_and_saved_adapter_prediction(tmp_path, tiny_local_qwen):
    from radar_vla.pipeline import load_models, predict_checkpoint, run_training
    from radar_vla.synthetic import write_synthetic_dataset
    torch.set_num_threads(1)
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=2)
    config = dict(model=dict(hidden_dim=16, num_heads=2, num_queries=4, max_agents=4, horizon_steps=6),
                  planner=dict(hidden_dim=16, num_heads=2, max_agents=4, horizon_steps=6),
                  language_model=dict(backend='hf', model_path=str(tiny_local_qwen),
                                      lora=True, dtype='float32', gradient_checkpointing=True))
    run_training(manifest, tmp_path / 'grounding', epochs=1, config=config)
    status = run_training(manifest, tmp_path / 'sft', stage='sft', epochs=1, config=config,
                          init_grounding=tmp_path / 'grounding/best.pt')
    assert status['state'] == 'complete'
    _, planner, state = load_models(tmp_path / 'sft/best.pt')
    assert state['planner']['format'] == 'radar_vla_hf_adapters_v2'
    assert planner.use_lora
    predictions = predict_checkpoint(tmp_path / 'sft/best.pt', manifest, tmp_path / 'predictions.jsonl')
    assert len(predictions) == 2
    for sample in predictions:
        assert torch.isfinite(torch.tensor(sample['risk'])).all()
        assert sample['plan']['valid']


def test_local_hf_lora_two_rank_cpu_ddp_training(tmp_path, tiny_local_qwen):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from radar_vla.pipeline import run_training
    from radar_vla.synthetic import write_synthetic_dataset

    torch.set_num_threads(1)
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=4)
    config = dict(model=dict(hidden_dim=16, num_heads=2, num_queries=4, max_agents=4, horizon_steps=6),
                  planner=dict(hidden_dim=16, num_heads=2, max_agents=4, horizon_steps=6),
                  language_model=dict(backend='hf', model_path=str(tiny_local_qwen),
                                      lora=True, dtype='float32', gradient_checkpointing=True))
    run_training(manifest, tmp_path / 'grounding', epochs=1, config=config)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    runner = tmp_path / 'run_ddp.py'
    runner.write_text(
        'import json, sys, torch\n'
        'from radar_vla.pipeline import run_training\n'
        'torch.set_num_threads(1)\n'
        'run_training(sys.argv[1], sys.argv[2], stage="sft", epochs=1, batch_size=1, accumulation_steps=2, '
        'device="cpu", config=json.load(open(sys.argv[3])), init_grounding=sys.argv[4])\n'
        'torch.distributed.destroy_process_group()\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(runner), str(manifest), str(tmp_path / 'sft'), str(config_path),
        str(tmp_path / 'grounding/best.pt')], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((tmp_path / 'sft/status.json').read_text())
    assert status['state'] == 'complete' and status['world_size'] == 2
    state = torch.load(tmp_path / 'sft/best.pt', weights_only=True)
    assert state['planner']['format'] == 'radar_vla_hf_adapters_v2'


def test_hf_without_risk_prefix_is_invariant_to_risk_condition(tiny_local_qwen):
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False).eval()
    planner.config.use_risk_token = False
    radar, risk, instructions = observations()
    targets = torch.tensor([[planner.tokenizer.token('<SHORT>'), planner.tokenizer.eos_id]] * 2)
    with torch.no_grad():
        before = planner(radar, risk, instructions, targets)
        after = planner(radar, risk + 100., instructions, targets)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_hf_instruction_over_budget_raises_instead_of_truncating(tiny_local_qwen):
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False)
    radar, risk, instructions = observations()
    planner.config.max_instruction_bytes = 8
    targets = torch.tensor([[planner.tokenizer.token('<SHORT>'), planner.tokenizer.eos_id]] * 2)
    with pytest.raises(ValueError, match='max_instruction_bytes'):
        planner(radar, risk, instructions, targets)
    # Budget is explicitly UTF-8 bytes, not Python character count.
    with pytest.raises(ValueError, match='max_instruction_bytes'):
        planner(radar, risk, ['向左转', 'safe'], targets)


def test_eval_policy_scoring_uses_checkpoint_recomputation_without_dropout(tiny_local_qwen, monkeypatch):
    from radar_vla.rl_policy import policy_statistics
    planner = make_planner(tiny_local_qwen, lora=True, gradient_checkpointing=True).eval()
    backbone = planner._native_model().base_model
    calls = []
    original = backbone._gradient_checkpointing_func
    def counted(function, *args, **kwargs):
        calls.append(True)
        assert not planner.training
        assert not backbone.layers[0].self_attn.training
        assert all(not module.training for module in planner.modules() if isinstance(module, torch.nn.Dropout))
        return original(function, *args, **kwargs)
    monkeypatch.setattr(backbone, '_gradient_checkpointing_func', counted)
    radar, risk, instructions = observations()
    # Mode is a genuine free choice and needs a nonzero score gradient.
    stats = policy_statistics(planner, radar[:1], risk[:1], instructions[:1],
                              [planner.tokenizer.token('<SHORT>')])
    (-stats['log_prob_sum']).backward()
    assert calls, 'configured HF gradient checkpointing was bypassed in eval scoring'
    assert not backbone.training
    assert planner.radar_projector.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('use_risk_token', [False, True])
def test_hf_cached_steps_match_full_qwen_with_padded_prefix(tiny_local_qwen, dtype, use_risk_token):
    """Wrong RoPE positions, masks or repeated prefix must change this real Qwen result."""
    planner = make_planner(tiny_local_qwen, lora=False, dtype=dtype,
                           gradient_checkpointing=False, generation_use_cache=True).eval()
    planner.config.use_risk_token = use_risk_token
    radar, risk, _ = observations()
    calls = []
    def observe(module, args, kwargs):
        calls.append((kwargs['inputs_embeds'].shape[1], kwargs.get('use_cache')))
    with torch.no_grad():
        prefix, mask = planner.encode_prefix(radar, risk, ['', 'Drive safely turn left'])
        decode = planner._generation_decoder(prefix, mask)
        inputs = torch.full((2, 1), planner.tokenizer.bos_id, dtype=torch.long)
        for token in ('<SHORT>', '<AGENTS>', '<END_AGENTS>', '<TRAJECTORY>'):
            expected = planner._decode_inputs(prefix, mask, inputs)[:, -1:]
            handle = planner._native_model().base_model.register_forward_pre_hook(observe, with_kwargs=True)
            try:
                actual = decode(inputs)
            finally:
                handle.remove()
            torch.testing.assert_close(actual, expected,
                                       atol=2e-6 if dtype == 'float32' else .003,
                                       rtol=1e-5 if dtype == 'float32' else .03)
            inputs = torch.cat((inputs, torch.full((2, 1), planner.tokenizer.token(token))), dim=1)
    assert calls == [(prefix.shape[1] + 1, True), (1, True), (1, True), (1, True)]


@pytest.mark.parametrize('mode,road_format', [('always_short', 'legacy'),
                                             ('always_long', 'legacy'),
                                             ('always_long', 'segments')])
def test_hf_cache_matches_greedy_and_forced_action_timing(tiny_local_qwen, mode, road_format):
    from radar_vla.execution import measure_plan
    torch.manual_seed(123)
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False,
                           generation_use_cache=False).eval()
    planner.config.reasoning_policy = mode
    planner.config.road_format = road_format
    planner.config.max_road_segments = 2
    radar, risk, _ = observations()
    instructions = ['', 'Drive safely turn left']
    expected = planner.generate(radar, risk, instructions, max_new_tokens=100)
    assert all(planner.decode(row, future_times_s=[.5, 1.])['valid'] for row in expected)
    calls = []
    def observe(module, args, kwargs):
        calls.append((kwargs['inputs_embeds'].shape[1], kwargs.get('use_cache')))
    planner.generation_use_cache = True
    handle = planner._native_model().base_model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        actual = planner.generate(radar, risk, instructions, max_new_tokens=100)
    finally:
        handle.remove()
    assert actual == expected
    assert sum(length > 1 for length, _ in calls) == len(instructions)
    assert len(calls) == sum(map(len, actual)) and all(cached for _, cached in calls)
    # Actual execution timing replay must use the same cached forward path.
    calls.clear()
    handle = planner._native_model().base_model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        timed = measure_plan(planner, radar[:1], risk[:1], instructions[:1],
                             ids=expected[0], max_new_tokens=100, future_times_s=[.5, 1.])
    finally:
        handle.remove()
    assert timed['ids'] == expected[0] and timed['plan']['valid']
    assert len(calls) == len(expected[0])
    assert all(length == 1 and cached for length, cached in calls[1:])
    assert calls[0][0] > 1 and calls[0][1]


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_hf_cache_sampling_rescores_with_uncached_lora_gradients(tiny_local_qwen, dtype):
    from radar_vla.rl_policy import sample_plan, policy_statistics, validate_policy_rescore
    torch.manual_seed(16)
    planner = make_planner(tiny_local_qwen, lora=True, dtype=dtype,
                           gradient_checkpointing=True, generation_use_cache=False).eval()
    radar, risk, instructions = observations()
    radar, risk, instructions = radar[:1], risk[:1], instructions[:1]
    torch.manual_seed(64)
    baseline = sample_plan(planner, radar, risk, instructions, max_new_tokens=100)
    planner.generation_use_cache = True
    torch.manual_seed(64)
    cached = sample_plan(planner, radar, risk, instructions, max_new_tokens=100)
    assert cached['ids'] == baseline['ids']
    assert cached['token_log_probs'] == pytest.approx(
        baseline['token_log_probs'], abs=2e-5 if dtype == 'float32' else .03)
    calls = []
    def observe(module, args, kwargs):
        calls.append(kwargs.get('use_cache'))
    handle = planner._native_model().base_model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        stats = policy_statistics(planner, radar, risk, instructions, cached['ids'])
        validate_policy_rescore(planner, cached, stats)
        (-stats['log_prob_sum']).backward()
    finally:
        handle.remove()
    assert calls and not any(calls)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in planner.named_parameters() if 'lora_' in name)
    assert not planner._native_model().base_model.training


def test_hf_cache_preserves_context_truncation_grammar_and_checkpoint(tiny_local_qwen):
    torch.manual_seed(42)
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=False,
                           generation_use_cache=False).eval()
    radar, risk, instructions = observations()
    with torch.no_grad():
        prefix, mask = planner.encode_prefix(radar, risk, instructions)
    planner.config.max_length = prefix.shape[1] + 5
    baseline = planner.generate(radar, risk, instructions, max_new_tokens=80)
    assert all(len(row) == 5 and not planner.decode(row)['valid'] for row in baseline)
    state = planner.checkpoint_state()
    planner.generation_use_cache = True
    assert planner.generate(radar, risk, instructions, max_new_tokens=80) == baseline
    planner.load_checkpoint_state(state)
    assert set(planner.checkpoint_state()['parameters']) == set(state['parameters'])
    with pytest.raises(ValueError, match='context'):
        planner._generate_ids(radar, risk, instructions, forced_ids=[baseline[0] + [0], baseline[1]])
    with pytest.raises(ValueError, match='grammar'):
        planner._generate_ids(radar, risk, instructions, forced_ids=[[planner.tokenizer.eos_id]] * 2)
    planner.config.max_length = prefix.shape[1]
    assert planner.generate(radar, risk, instructions) == [[], []]


def test_hf_cache_requires_explicit_boolean_and_eval_generation(tiny_local_qwen):
    # A string "false" from a malformed config must not silently enable caching.
    with pytest.raises(ValueError, match='generation_use_cache'):
        make_planner(tiny_local_qwen, lora=False, generation_use_cache='false')
    planner = make_planner(tiny_local_qwen, lora=False, gradient_checkpointing=True,
                           generation_use_cache=True)
    radar, risk, instructions = observations()
    with pytest.raises(ValueError, match='eval'):
        planner.generate(radar, risk, instructions, max_new_tokens=3)
