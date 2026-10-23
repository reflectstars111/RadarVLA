"""Resume only sensor validation after a hash-matched motion acceptance PASS."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from radar_vla.agent50_migration import sha256
from radar_vla.moving_migration import write_json
from radar_vla.moving_agents import POLICY


def check_acceptance(output, config):
    output, config = Path(output), Path(config)
    acceptance = json.loads((output/'acceptance.json').read_text())
    provenance = json.loads((output/'provenance.json').read_text())
    migration = json.loads((output/'migration.json').read_text())
    if (acceptance.get('status') != 'PASS' or acceptance.get('policy') != POLICY
            or acceptance.get('raw_evidence_checked') is not True
            or not acceptance.get('representative_gradients')
            or not (output/'raw_evidence.json').is_file()):
        raise ValueError('Full motion acceptance PASS is required before finalizing')
    digest = sha256(output/'manifest.jsonl')
    if digest != acceptance['manifest_sha256'] or digest != migration['manifest_sha256']:
        raise ValueError('Manifest changed since full acceptance')
    source_digest = sha256(Path(provenance['source']))
    if source_digest != acceptance['source_sha256'] or source_digest != provenance['source_sha256']:
        raise ValueError('Source changed since full acceptance')
    if sha256(config) != provenance['config_sha256']:
        raise ValueError('Configuration changed since full acceptance')
    for path, expected in provenance['code'].items():
        if sha256(path) != expected:
            raise ValueError('Migration implementation changed: ' + path)
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = root/'prepared/taskwise_moving_v1'
    reports = root/'analysis/taskwise_moving_v1'
    project = Path(__file__).resolve().parents[1]
    with (reports/'migration.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(reports/'status.json', dict(status='VERIFYING', phase='acceptance_hash_check', pid=os.getpid()))
            acceptance = check_acceptance(output, args.config)
            command = ['bash', str(project/'scripts/radarvla.sh'), 'validate', '--manifest', str(output/'manifest.jsonl')]
            write_json(reports/'finalization.json', dict(
                entrypoint_sha256=sha256(Path(__file__)), command=command,
                acceptance_sha256=sha256(output/'acceptance.json'), reused_raw_and_gradient_acceptance=True))
            print('Full raw-evidence/Risk/gradient acceptance PASS and hashes confirmed; validating all sensor arrays', flush=True)
            write_json(reports/'status.json', dict(status='VERIFYING', phase='sensor_validation', pid=os.getpid()))
            subprocess.run(command, check=True, cwd=project)
            validation = json.loads((output/'validation.json').read_text())
            expected = acceptance['manifest_sha256']
            if validation.get('status') != 'PASS' or validation.get('manifest_sha256') != expected or sha256(output/'manifest.jsonl') != expected:
                raise ValueError('Final validation has no matching PASS manifest hash')
            write_json(reports/'status.json', dict(status='COMPLETE', pid=os.getpid(),
                manifest=str(output/'manifest.jsonl'), manifest_sha256=expected))
            print('COMPLETE: full motion acceptance and full sensor validation passed', flush=True)
        except BaseException as exc:
            write_json(reports/'status.json', dict(status='FAILED', pid=os.getpid(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    main()
