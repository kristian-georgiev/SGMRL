"""Path hack to make tests work."""

import os
import sys

class context:
    def __enter__(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)

    def __exit__(self, *args):
        pass
