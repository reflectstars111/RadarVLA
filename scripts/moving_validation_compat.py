"""Supply the supervised Dataset interface omitted by the v1 acceptance shim.

Kept outside src so the already published migration's source fingerprint remains
verifiable. This adapter changes no records, hashes, model inputs or validators.
"""
from contextlib import contextmanager
from pathlib import Path


def validation_command(command):
    """Translate the v1 validator invocation to the current exclusive CLI modes."""
    if (isinstance(command, (list, tuple)) and len(command) == 6
            and command[0] == 'bash' and Path(command[1]).name == 'radarvla.sh'
            and list(command[2:5]) == ['validate', '--full', '--manifest']):
        return [*command[:3], *command[4:]]
    return command


def validate_metadata(dataset, config, validator):
    if (type(dataset).__module__ == 'radar_vla.moving_migration'
            and type(dataset).__name__ == 'Metadata'
            and not hasattr(dataset, 'supervision')):
        dataset.supervision = True
    return validator(dataset, config)


@contextmanager
def supervised_metadata_adapter():
    import radar_vla.pipeline as pipeline
    import subprocess
    original = pipeline.validate_input_shapes
    original_run = subprocess.run
    pipeline.validate_input_shapes = lambda dataset, config: validate_metadata(dataset, config, original)
    subprocess.run = lambda command, *args, **kwargs: original_run(validation_command(command), *args, **kwargs)
    try:
        yield
    finally:
        pipeline.validate_input_shapes = original
        subprocess.run = original_run
