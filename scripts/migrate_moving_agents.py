"""Migrate, validate and report motion-only data without starting training."""
import argparse
from pathlib import Path

from radar_vla.agent50_migration import sha256
from radar_vla.moving_migration import main, write_json
from moving_validation_compat import supervised_metadata_adapter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root', type=Path)
    args, _ = parser.parse_known_args()
    if args.data_root is not None:
        reports = args.data_root/'analysis/taskwise_moving_v1'
        reports.mkdir(parents=True, exist_ok=True)
        write_json(reports/'verification_adapter.json', dict(
            reason='v1 acceptance Metadata omitted supervision=True; v1 final validator used mutually exclusive --full and --manifest',
            entrypoint=str(Path(__file__).resolve()), entrypoint_sha256=sha256(Path(__file__)),
            adapter_sha256=sha256(Path(__file__).with_name('moving_validation_compat.py')),
            original_migration_provenance_preserved=True))
        print('Acceptance adapter: supply Metadata.supervision=True; all original hash checks retained', flush=True)
    with supervised_metadata_adapter():
        main()
