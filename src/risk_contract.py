"""Physical horizons, kept separate from trajectory output knot counts."""
LEGACY_HORIZONS = '1_2_3_v1'
SHORT_HORIZONS = '0p5_1_2_v2'
LOCAL_SHORT_SCHEMA = 'local_agent_collision_0p5_1_2_v2'


def contract(schema=LEGACY_HORIZONS):
    if schema == LEGACY_HORIZONS:return (1.,2.,3.),3.
    if schema == SHORT_HORIZONS:return (.5,1.,2.),2.
    raise ValueError(f'Unknown risk horizon schema: {schema}')


def horizon_schema(risk_supervision_schema):
    return SHORT_HORIZONS if risk_supervision_schema == LOCAL_SHORT_SCHEMA else LEGACY_HORIZONS


def horizon_keys(schema=LEGACY_HORIZONS):return [f'{h:g}s' for h in contract(schema)[0]]


def validate_batch(batch, schema):
    expected = int(schema == SHORT_HORIZONS)
    found = batch.get('risk_horizon_schema_id')
    if found is None:
        if expected:raise ValueError('New Risk horizons require explicit batch metadata')
    elif not bool((found == expected).all()):
        raise ValueError('Batch and model Risk horizons differ; no implicit relabeling')
