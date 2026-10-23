"""Apply the pedestrian/cyclist ignore policy to a completed state migration."""
import argparse
from pathlib import Path
from radar_vla.agent50_filter import filter_manifest

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    filter_manifest(parser.parse_args().directory)
