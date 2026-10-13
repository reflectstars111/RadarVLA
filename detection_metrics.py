"""Thresholded detection metrics with explicit FP/FN and category coverage."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .categories import DETECTION_CLASSES


class DetectionMetrics:
    def __init__(self, match_distance_m=2.):
        if not np.isfinite(match_distance_m) or match_distance_m <= 0:
            raise ValueError('match_distance_m must be finite and positive')
        self.distance = match_distance_m
        self.samples = self.skipped = self.truncated = 0
        self.tp = self.fp = self.fn = 0
        self.count_errors, self.vehicle_errors, self.positions, self.ade, self.fde = [], [], [], [], []
        self.classes = {name: dict(tp=0, fp=0, fn=0) for name in DETECTION_CLASSES}
        self.known_classes = self.unknown_classes = 0

    def update(self, rows, batch):
        def array(key):
            return batch[key].detach().cpu().numpy()

        active, truth = array('agent_mask').astype(bool), array('agent_state')
        known = array('agent_supervision_mask').astype(bool) if 'agent_supervision_mask' in batch else np.ones(len(active), bool)
        labels = array('agent_class') if 'agent_class' in batch else np.full(active.shape, -1)
        class_mask = array('agent_class_mask').astype(bool) if 'agent_class_mask' in batch else np.zeros_like(active)
        futures, future_mask = array('agent_future'), array('agent_future_mask').astype(bool)
        if len(rows) != len(active):
            raise ValueError('Detection rows must match batch size')
        for b, row in enumerate(rows):
            if not known[b]:
                self.skipped += 1
                continue
            ids = np.flatnonzero(active[b])
            if not np.isfinite(truth[b, ids, :2]).all():
                raise ValueError('Detection metrics need known target positions')
            self.samples += 1
            self.truncated += int(row['detection_truncated'])
            detected = row['detections']
            self.count_errors.append(abs(len(detected) - len(ids)))
            if class_mask[b, ids].all():
                self.vehicle_errors.append(abs(row['num_vehicles'] - int(np.sum(labels[b, ids] == 0))))
            self.known_classes += int(class_mask[b, ids].sum())
            self.unknown_classes += int((~class_mask[b, ids]).sum())
            pairs = []
            if detected and len(ids):
                xy = np.asarray([d['position_m'] for d in detected])
                distances = np.linalg.norm(xy[:, None] - truth[b, ids, :2][None], axis=-1)
                # Large penalties maximize the number of admissible one-to-one
                # matches before minimizing localization error.
                cost = np.where(distances <= self.distance, distances,
                                (len(detected)+len(ids)+1) * (self.distance+1))
                pi, gi = linear_sum_assignment(cost)
                pairs = [(int(p), int(ids[g])) for p, g in zip(pi, gi) if distances[p, g] <= self.distance]
            self.tp += len(pairs)
            self.fp += len(detected) - len(pairs)
            self.fn += len(ids) - len(pairs)
            matched_p, matched_g = {p for p, _ in pairs}, {g for _, g in pairs}
            for p, g in pairs:
                d = detected[p]
                self.positions.append(float(np.linalg.norm(np.asarray(d['position_m'])-truth[b, g, :2])))
                pred_future = np.asarray(d['future_xy'])
                if pred_future.shape != futures[b, g].shape:
                    raise ValueError('Detection and GT future horizons differ')
                mask = future_mask[b, g] & np.isfinite(futures[b, g]).all(-1)
                error = np.linalg.norm(pred_future - futures[b, g], axis=-1)
                self.ade.extend(error[mask].tolist())
                if mask[-1]:
                    self.fde.append(float(error[-1]))
                # An unknown semantic GT cannot mark a predicted class right or wrong.
                if class_mask[b, g]:
                    label, prediction = DETECTION_CLASSES[int(labels[b, g])], d['category']
                    if prediction == label:
                        self.classes[label]['tp'] += 1
                    else:
                        self.classes[label]['fn'] += 1
                        if prediction in self.classes:
                            self.classes[prediction]['fp'] += 1
            for p, d in enumerate(detected):
                if p not in matched_p and d['category'] in self.classes:
                    self.classes[d['category']]['fp'] += 1
            for g in ids:
                if g not in matched_g and class_mask[b, g]:
                    self.classes[DETECTION_CLASSES[int(labels[b, g])]]['fn'] += 1

    def compute(self):
        def rates(tp, fp, fn):
            return dict(true_positives=tp, false_positives=fp, false_negatives=fn,
                        precision=tp/(tp+fp) if tp+fp else None,
                        recall=tp/(tp+fn) if tp+fn else None,
                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

        def mean(values):
            return float(np.mean(values)) if values else None

        return dict(samples=self.samples, unannotated_samples=self.skipped,
                    truncated_samples=self.truncated, match_distance_m=self.distance,
                    **rates(self.tp, self.fp, self.fn), count_mae=mean(self.count_errors),
                    count_exact_accuracy=mean([x == 0 for x in self.count_errors]),
                    vehicle_count_mae=mean(self.vehicle_errors), vehicle_count_samples=len(self.vehicle_errors),
                    vehicle_count_exact_accuracy=mean([x == 0 for x in self.vehicle_errors]),
                    matched_center_error_m=mean(self.positions), matched_objects=len(self.positions),
                    matched_trajectory_ade_m=mean(self.ade), matched_trajectory_points=len(self.ade),
                    matched_trajectory_fde_m=mean(self.fde), matched_trajectory_fde_objects=len(self.fde),
                    category_labeled_objects=self.known_classes, category_unknown_objects=self.unknown_classes,
                    per_class={name: rates(**values) for name, values in self.classes.items()},
                    semantics='confidence/NMS-filtered detections; class-agnostic center matching, then class scoring; not AP; trajectory errors conditional on detection matches')
