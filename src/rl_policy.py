"""Grammar-constrained on-policy sampling shared by tiny and HF planners."""
from __future__ import annotations

import math
import torch


def _validate(planner, radar, temperature):
    if getattr(planner.config, 'action_representation', 'token_spline_v1') != 'token_spline_v1':
        raise ValueError('continuous action RL requires a native continuous policy; legacy token log-probabilities are unsupported')
    if planner.training:
        raise ValueError('RL policies must use eval mode for reproducible dropout-free probabilities')
    if radar.shape[0] != 1:
        raise ValueError('Response latency is measured per event; use batch size one')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')


@torch.no_grad()
def sample_plan(planner, radar, risk, instructions, max_new_tokens=1024,
                temperature=1., greedy=False):
    """Sample the actual deployed grammar, including mode and agent count.

    Forced tokens contribute zero log probability. Eval mode disables dropout,
    not autograd in the separate teacher-forced rescoring function.
    """
    _validate(planner, radar, temperature)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    log_probs = []
    def choose(logits, allowed):
        distribution = torch.distributions.Categorical(logits=logits[allowed].float() / temperature)
        index = distribution.logits.argmax() if greedy else distribution.sample()
        log_probs.append(float(distribution.log_prob(index)))
        return allowed[int(index)]
    ids = planner._generate_ids(radar, risk, instructions, max_new_tokens,
                                token_selector=choose)[0]
    if not ids:
        raise ValueError('No generation context remains after the observation prefix')
    return dict(ids=ids, log_prob_sum=sum(log_probs), token_log_probs=log_probs)


def policy_statistics(planner, radar, risk, instructions, ids, temperature=1., reference=None):
    """Sum of action log probabilities and exact conditional KL on sampled prefixes.

    The KL averages only genuine choices; grammar punctuation cannot dilute it.
    This regularizer anchors the SFT distribution; it is not a claim to compute
    the exact KL between all possible complete variable-length sequences.
    """
    _validate(planner, radar, temperature)
    if not ids:
        raise ValueError('Cannot score an empty action')
    allowed_rows = []
    for i, token in enumerate(ids):
        if i and ids[i - 1] == planner.tokenizer.eos_id:
            raise ValueError('Tokens after EOS violate the grammar')
        allowed = planner._allowed_next(ids[:i])
        if token not in allowed:
            raise ValueError('Action violates the policy grammar')
        allowed_rows.append(allowed)
    targets = torch.tensor([ids], dtype=torch.long, device=radar.device)
    logits = planner(radar, risk, instructions, targets)[0].float()
    reference_logits = None
    if reference is not None:
        _validate(reference, radar, temperature)
        with torch.no_grad():
            reference_logits = reference(radar, risk, instructions, targets)[0].float()
    log_prob = logits.sum() * 0.
    token_log_probs = []
    kl, decisions = log_prob, 0
    for i, allowed in enumerate(allowed_rows):
        if len(allowed) == 1:
            token_log_probs.append(logits[i].sum() * 0.)
            continue
        decisions += 1
        logp = (logits[i, allowed] / temperature).log_softmax(-1)
        selected = logp[allowed.index(ids[i])]
        token_log_probs.append(selected)
        log_prob = log_prob + selected
        if reference_logits is not None:
            logq = (reference_logits[i, allowed] / temperature).log_softmax(-1)
            kl = kl + (logp.exp() * (logp - logq)).sum()
    return dict(log_prob_sum=log_prob, token_log_probs=torch.stack(token_log_probs),
                kl=kl / max(decisions, 1), decisions=decisions)


def validate_policy_rescore(planner, sampled, statistics):
    """Compare per choice, without accumulating BF16 rounding over long outputs.

    Growing-prefix decode and batched teacher forcing can use different GEMM
    shapes/attention kernels. BF16 needs a larger numerical tolerance than FP32;
    this is a drift assertion, not importance weighting or PPO clipping.
    """
    dtype = (planner._native_model().get_input_embeddings().weight.dtype
             if hasattr(planner, '_native_model') else next(planner.parameters()).dtype)
    low_precision = dtype in (torch.bfloat16, torch.float16)
    old = torch.tensor(sampled['token_log_probs'], dtype=torch.float32)
    new = statistics['token_log_probs'].detach().float().cpu()
    if (old.shape != new.shape or not torch.isfinite(new).all()
            or not torch.allclose(old, new, atol=.03 if low_precision else 2e-4,
                                  rtol=.005 if low_precision else 1e-5)):
        raise ValueError('Sampling/rescoring policy probabilities differ beyond numerical tolerance')


def leave_one_out_advantages(rewards):
    """Independent same-observation baselines; no reward scaling or clipping."""
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError('At least two finite rewards are needed per event')
    total = sum(rewards)
    return [reward - (total - reward) / (len(rewards) - 1) for reward in rewards]
