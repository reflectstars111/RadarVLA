"""Create an episode-disjoint, stratified manifest without copying sensor arrays."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


SPLITS = ('train', 'val', 'test')
TARGET = {'train': .70, 'val': .15, 'test': .15}


def episode_table(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['episode_id']].append(row)
    features = {}
    for episode, rows in groups.items():
        positive = any(row['risk_label']['pcol_mask'][2] and
                       row['risk_label']['pcol'][2] == 1 for row in rows)
        features[episode] = dict(rows=len(rows), positive=int(positive),
                                 category=Counter(row['category'] for row in rows),
                                 weather=Counter(row['weather'] for row in rows),
                                 instruction=Counter(row['language']['instruction'] for row in rows))
    return groups, features


def choose_splits(features, seed=42, starts=40):
    episodes = sorted(features)
    totals = Counter()
    distinct = {field: Counter() for field in ('category', 'weather', 'instruction')}
    substantial = Counter()
    for value in features.values():
        totals['rows'] += value['rows']
        totals['episodes'] += 1
        totals['positive'] += value['positive']
        for field in distinct:
            distinct[field].update(value[field])
        if value['rows'] >= 20:
            substantial.update(value['category'])

    def cost(assignment):
        bins = {split: {'rows': 0, 'episodes': 0, 'positive': 0,
                        'substantial': Counter(),
                        **{field: Counter() for field in distinct}} for split in SPLITS}
        for episode, split in assignment.items():
            value = features[episode]
            bucket = bins[split]
            bucket['rows'] += value['rows']
            bucket['episodes'] += 1
            bucket['positive'] += value['positive']
            if value['rows'] >= 20:
                bucket['substantial'].update(value['category'])
            for field in distinct:
                bucket[field].update(value[field])
        score = 0.
        for split in SPLITS:
            target = TARGET[split]
            bucket = bins[split]
            score += 25 * abs(bucket['rows'] / totals['rows'] - target)
            score += 8 * abs(bucket['episodes'] / totals['episodes'] - target)
            score += 6 * abs(bucket['positive'] / max(1, totals['positive']) - target)
            for field, labels in distinct.items():
                for label, count in labels.items():
                    if count:
                        score += (3 if field == 'category' else 1) * abs(
                            bucket[field][label] / count - target) / len(labels)
            if split != 'train':
                score += 100 * max(0, 2 - bucket['positive'])
                for label, count in substantial.items():
                    if count >= 3 and bucket['substantial'][label] == 0:
                        score += 100
                if split == 'val' and substantial['scenario7_9'] >= 2 and bucket['substantial']['scenario7_9'] == 0:
                    score += 150
                for label in distinct['weather']:
                    score += 100 * (bucket['weather'][label] < 30)
                for label in distinct['instruction']:
                    score += 50 * (bucket['instruction'][label] < 20)
            else:
                score += 200 * sum(bucket['substantial'][label] == 0 for label in substantial)
        return score

    fixed = {episode: 'train' for episode in episodes
             if features[episode]['category'].get('scenario4')}
    free = [episode for episode in episodes if episode not in fixed]
    rng = random.Random(seed)
    best, best_score = None, float('inf')
    for _ in range(starts):
        order = free[:]
        rng.shuffle(order)
        assignment = dict(fixed)
        for episode in order:
            options = []
            for split in SPLITS:
                trial = dict(assignment, **{episode: split})
                # Full-assignment costs are defined for partial assignments too.
                options.append((cost(trial) + rng.random() * .05, split))
            assignment[episode] = min(options)[1]
        for _ in range(8):
            improved = False
            rng.shuffle(order)
            for episode in order:
                old = assignment[episode]
                current = cost(assignment)
                for split in SPLITS:
                    if split == old:
                        continue
                    assignment[episode] = split
                    candidate = cost(assignment)
                    if candidate + 1e-9 < current:
                        old, current, improved = split, candidate, True
                    else:
                        assignment[episode] = old
            if not improved:
                break
        score = cost(assignment)
        if score < best_score:
            best, best_score = assignment.copy(), score
    return best, best_score


def _resolve_path(source, value):
    return str((source.parent / value).resolve()) if isinstance(value, str) else value


def rebase_paths(row, source):
    row = dict(row)
    sensors = dict(row['sensors'])
    radar = dict(sensors['radar'])
    for field in ('power', 'unfolded_doppler', 'doppler_valid'):
        if field in radar:
            value = radar[field]
            radar[field] = ([_resolve_path(source, p) for p in value]
                            if isinstance(value, list) else _resolve_path(source, value))
    sensors['radar'] = radar
    if 'camera' in sensors:
        sensors['camera'] = {key: {**value, 'path': _resolve_path(source, value['path'])}
                             for key, value in sensors['camera'].items()}
    if 'lidar' in sensors:
        sensors['lidar'] = {key: _resolve_path(source, value)
                            for key, value in sensors['lidar'].items()}
    row['sensors'] = sensors
    if row.get('route_archive'):
        row['route_archive'] = _resolve_path(source, row['route_archive'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if output.exists():
        raise ValueError('output directory already exists')
    records = [json.loads(line) for line in source.open() if line.strip()]
    groups, features = episode_table(records)
    assignment, score = choose_splits(features, args.seed)
    if set(assignment) != set(groups):
        raise RuntimeError('some episodes were not assigned')
    output.mkdir(parents=True)
    manifest = output / 'manifest.jsonl'
    with manifest.open('w') as stream:
        for original in records:
            row = rebase_paths(original, source)
            row['split'] = assignment[row['episode_id']]
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    split_stats = {}
    for split in SPLITS:
        ids = [episode for episode, value in assignment.items() if value == split]
        entries = [row for row in records if assignment[row['episode_id']] == split]
        split_stats[split] = dict(episodes=len(ids), rows=len(entries),
                                  positive_episodes=sum(features[episode]['positive'] for episode in ids),
                                  positive_frames=sum(row['risk_label']['pcol_mask'][2] and
                                                      row['risk_label']['pcol'][2] == 1 for row in entries),
                                  category=dict(Counter(row['category'] for row in entries)),
                                  weather=dict(Counter(row['weather'] for row in entries)),
                                  instruction=dict(Counter(row['language']['instruction'] for row in entries)),
                                  episode_ids=sorted(ids))
    if (not 0.67 <= split_stats['train']['rows'] / len(records) <= 0.73
            or any(not 0.12 <= split_stats[split]['rows'] / len(records) <= 0.18
                   for split in ('val', 'test'))):
        raise RuntimeError('split frame ratios fell outside target tolerance')
    if any(split_stats[split]['positive_episodes'] < 2 for split in ('val', 'test')):
        raise RuntimeError('validation or test has fewer than two positive episodes')
    if not split_stats['train']['category'].get('scenario7_9') or not split_stats['val']['category'].get('scenario7_9'):
        raise RuntimeError('scenario7_9 must remain in train and validation')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = dict(source=str(source), manifest=str(manifest), manifest_sha256=digest,
                  seed=args.seed, objective=score, target=TARGET, splits=split_stats,
                  scenario4_gap='Only one substantial scenario4 episode; kept in train')
    (output / 'split_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: {field: value for field, value in stats.items() if field != 'episode_ids'}
                      for key, stats in split_stats.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
