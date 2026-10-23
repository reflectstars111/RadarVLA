"""Risk-head-only Oracle mode distillation; no future labels in student inputs."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import loss_grounding


def is_risk_parameter(name: str) -> bool:
    return name == 'risk_query' or name.startswith(('risk_attention.', 'risk_head.'))


def configure_risk_head_training(grounder, planner):
    """Freeze encoder/detector/ego and every Planner parameter, including LoRA."""
    if not grounder.config.planner_ego_context or not grounder.config.use_ego:
        raise ValueError('Cached risk refinement requires the explicit shared Ego context')
    if (not planner.config.use_risk_token or planner.config.risk_source != 'predicted'
            or planner.config.risk_projection != 'none'):
        raise ValueError('Refinement requires the original predicted Risk condition with projection none')
    grounder.eval()
    planner.eval()
    for name, parameter in grounder.named_parameters():
        parameter.requires_grad_(is_risk_parameter(name))
        parameter.grad = None
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return [parameter for parameter in grounder.parameters() if parameter.requires_grad]


def risk_from_frozen_tokens(grounder, planner_tokens: Tensor):
    """Reuse the canonical head on frozen sensor features, not a second model."""
    if (not grounder.config.planner_ego_context or not grounder.config.use_ego
            or planner_tokens.ndim != 3
            or planner_tokens.shape[1:] != (grounder.config.num_queries + 1, grounder.config.hidden_dim)):
        raise ValueError('Expected [B,num_queries+1,C] Radar tokens followed by the shared Ego token')
    tokens = planner_tokens.detach()
    return grounder.risk_from_context(tokens[:, :-1], tokens[:, -1])


def first_mode_logits(planner, radar_tokens: Tensor, risk: Tensor, instructions: list[str]) -> Tensor:
    """BOS readout in [SHORT,LONG] order; no teacher text enters this forward."""
    short, long = planner.tokenizer.token('<SHORT>'), planner.tokenizer.token('<LONG>')
    placeholder = torch.full((len(instructions), 1), short, dtype=torch.long, device=risk.device)
    return planner(radar_tokens.detach(), risk, instructions, placeholder)[:, 0, [short, long]].float()


def oracle_mode_kl(student: Tensor, teacher: Tensor, known: Tensor, temperature: float = 1.) -> Tensor:
    """KL(teacher || student) on complete-label rows, with a stopped teacher."""
    if student.ndim != 2 or student.shape[-1] != 2 or teacher.shape != student.shape:
        raise ValueError('student and teacher logits must have shape [B,2]')
    if known.shape != student.shape[:1] or known.dtype != torch.bool:
        raise ValueError('known must be a boolean [B] mask')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher[known]).all()):
        raise ValueError('Student and selected teacher logits must be finite')
    if not bool(known.any()):
        return student.sum() * 0
    target = F.softmax(teacher[known].detach().float() / temperature, -1)
    predicted = F.log_softmax(student[known].float() / temperature, -1)
    return F.kl_div(predicted, target, reduction='batchmean') * temperature**2


def refinement_loss(grounder, planner, batch: dict, mode_weight: float = 1., temperature: float = 1.):
    """Existing physical loss plus mode-only KL; no Oracle trajectory targets."""
    if not math.isfinite(mode_weight) or mode_weight < 0:
        raise ValueError('mode_weight must be finite and nonnegative')
    prediction = risk_from_frozen_tokens(grounder, batch['radar_tokens'])
    physical = loss_grounding(prediction, dict(risk_target=batch['risk_target'], risk_mask=batch['risk_mask']))
    kl = prediction['risk'].sum() * 0
    if mode_weight:
        student = first_mode_logits(planner, batch['radar_tokens'], prediction['risk'], batch['instruction'])
        known = batch['risk_mask'].all(-1) & torch.isfinite(batch['risk_target']).all(-1)
        kl = oracle_mode_kl(student, batch['oracle_mode_logits'], known, temperature)
    return dict(total=physical['total'] + mode_weight * kl, physical=physical['total'],
                mode_kl=kl, collision=physical['collision'], distance=physical['distance'],
                brake=physical['brake'], monotonicity=physical['monotonicity'])
