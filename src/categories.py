"""Explicit semantic labels; missing labels never mean vehicle/background."""

DETECTION_CLASSES = ('vehicle', 'pedestrian', 'cyclist', 'other')
ALIASES = {
    'vehicle': 'vehicle', 'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle',
    'van': 'vehicle', 'motorcycle': 'vehicle',
    'pedestrian': 'pedestrian', 'person': 'pedestrian',
    'cyclist': 'cyclist', 'bicycle': 'cyclist', 'bicyclist': 'cyclist',
    'other': 'other', 'unknown': 'unknown',
}


def category_name(agent):
    names = []
    for key in ('category', 'class_name', 'class', 'type'):
        value = agent.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or value.strip().lower() not in ALIASES:
            raise ValueError(f'Unsupported agent category {value!r}; map labels to {DETECTION_CLASSES} or unknown')
        names.append(ALIASES[value.strip().lower()])
    if len(set(names)) > 1:
        raise ValueError('Conflicting agent category fields')
    return names[0] if names else 'unknown'
