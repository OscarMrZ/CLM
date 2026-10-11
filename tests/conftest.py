import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "train"), os.path.join(REPO, "preprocessing"),
          os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
