#!/usr/bin/env python3
"""Entry point for the 2D inspection sandbox.

Runs standalone (no ROS):  ``python sim2d/run.py``

Optional shape override:    ``python sim2d/run.py --shape circle``
"""

import argparse
import os
import sys

# Allow `python sim2d/run.py` by putting the package parent on the path, so the
# relative imports inside the package resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim2d.shapes import SHAPES          # noqa: E402
from sim2d.viz import SimApp             # noqa: E402
from sim2d.world import World            # noqa: E402

import matplotlib.pyplot as plt          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="2D inspection control sandbox")
    ap.add_argument("--shape", default="sine", choices=sorted(SHAPES),
                    help="initial surface shape")
    args = ap.parse_args()

    app = SimApp(World(shape_name=args.shape))
    plt.show()
    return app


if __name__ == "__main__":
    main()
