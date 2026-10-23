"""Explicit semantic labels; missing labels never mean vehicle/background."""

DETECTION_CLASSES = ('vehicle', 'pedestrian', 'cyclist', 'other')
ALIASES = {
    'vehicle': 'vehicle', 'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle',
    'van': 'vehicle', 'motorcycle': 'vehicle',
    'pedestrian': 'pedestrian', 'person': 'pedestrian',
    'cyclist': 'cyclist', 'bicycle': 'cyclist', 'bicyclist': 'cyclist',
    'other': 'other', 'unknown': 'unknown', 'agent': 'agent',
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


LEGACY_CLASS_SCHEMA = 'legacy4_v1'
AGENT_CLASS_SCHEMA = 'agent_v1'
AGENT_LABEL_SCHEMA = 'agent50_v1'


def detection_classes(schema=LEGACY_CLASS_SCHEMA):
    if schema == LEGACY_CLASS_SCHEMA:
        return DETECTION_CLASSES
    if schema == AGENT_CLASS_SCHEMA:
        return ('agent',)
    raise ValueError(f'Unsupported detection_class_schema {schema!r}')


def record_class_schema(record):
    schema = record.get('grounding_label_schema')
    if schema is None:
        return LEGACY_CLASS_SCHEMA
    if schema == AGENT_LABEL_SCHEMA:
        return AGENT_CLASS_SCHEMA
    raise ValueError(f'Unsupported grounding_label_schema {schema!r}')
