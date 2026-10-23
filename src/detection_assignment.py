"""Detached local one-to-one detector assignment with exact rotated-box IoU."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .detection import rotated_iou


@torch.no_grad()
def local_quality_assignment(anchors, predicted_xy, target_xy, grid_shape, eligible_indices, *,
                             anchor_cost, predicted_size=None, predicted_heading=None,
                             target_size=None, target_heading=None, target_box_mask=None):
    """Return rows within ``eligible_indices``, target columns and fallback flag.

    The nearest fixed anchor determines each GT's cell. Its clipped 3x3 grid
    neighbourhood includes every slot, intersected with the caller's eligibility
    mask. Costs are centre error / 2 metres plus (1 - exact rotated IoU) when
    box annotation is known; unknown boxes use centre error only. Neither
    objectness nor motion/future predictions participate in this decision.

    If the entire local bipartite graph has no complete one-to-one matching,
    use the original global anchor cost for the whole frame, retaining all GT.
    ``anchor_cost`` is passed unchanged, retaining the caller's canonical target
    order while reverting to the legacy anchor-distance cost.
    """
    if grid_shape is None:
        raise ValueError('local_quality_v1 requires detection grid shape metadata')
    dimensions = grid_shape.detach().cpu().tolist() if isinstance(grid_shape, torch.Tensor) else list(grid_shape)
    if len(dimensions) != 2 or any(type(n) is not int or n < 1 for n in dimensions):
        raise ValueError('detection grid shape must contain two positive integers')
    height, width = dimensions
    cells = height * width
    if anchors.ndim != 2 or anchors.shape[1] != 2 or not len(anchors) or len(anchors) % cells:
        raise ValueError('detection grid shape must divide the dense anchor count')
    slots = len(anchors) // cells
    grouped = anchors.reshape(cells, slots, 2)
    if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
        raise ValueError('detection grid slots must share the same fixed cell anchor')
    if (predicted_xy.shape != anchors.shape or target_xy.ndim != 2 or target_xy.shape[1] != 2
            or not torch.isfinite(anchors).all() or not torch.isfinite(predicted_xy[eligible_indices]).all()
            or not torch.isfinite(target_xy).all()):
        raise ValueError('quality assignment requires finite anchor, prediction and target centres')
    # Use every fixed cell when locating the GT, including cells outside the
    # eligible sector. Eligibility does not move a target's physical cell.
    nearest = torch.cdist(anchors.float(), target_xy.float()).argmin(dim=0) // slots
    candidate_cells = eligible_indices // slots
    local = (((candidate_cells[:, None] // width - nearest[None] // width).abs() <= 1)
             & ((candidate_cells[:, None] % width - nearest[None] % width).abs() <= 1))
    local = local.cpu().numpy()
    centres = torch.cdist(predicted_xy[eligible_indices].float(), target_xy.float()).cpu().numpy()
    quality = centres.astype(np.float64) / 2.
    if target_box_mask is not None and bool(target_box_mask.any()):
        if any(value is None for value in (predicted_size, predicted_heading, target_size, target_heading)):
            raise ValueError('Known quality-assignment boxes require sizes and headings')
        prediction_boxes = torch.cat((predicted_xy[eligible_indices], predicted_size[eligible_indices],
                                      predicted_heading[eligible_indices, None]), -1).float().cpu().numpy()
        target_boxes = torch.cat((target_xy, target_size, target_heading[:, None]), -1).float().cpu().numpy()
        known = target_box_mask.bool().cpu().numpy()
        for row, column in np.argwhere(local & known[None]):
            quality[row, column] += 1. - rotated_iou(prediction_boxes[row], target_boxes[column])
    try:
        rows, columns = linear_sum_assignment(np.where(local, quality, np.inf))
    except ValueError:
        # Geometry is already finite, so the inf-masked graph is infeasible.
        rows, columns = linear_sum_assignment(anchor_cost.detach().float().cpu().numpy())
        return rows, columns, True
    if len(columns) != len(target_xy):
        raise ValueError('Insufficient dense candidates to supervise every target')
    return rows, columns, False
