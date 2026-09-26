"""Scheduler for the 2001-2007 dance archive batch run.

See the design notes, which are not part of this tree
"""

import os

# From the environment, under the names tools/archive-batch-manifest.sh already
# honours. Hardcoded, every run on every machine publishes into gpu1's live
# 2.5 TB archive -- so previewing a change means previewing it on the real
# tree, and a test run has nowhere else to put its output.
ARCHIVE_ROOT = os.environ.get("ARCHIVE_ROOT", "/mnt/media/dance")
ENCODED_SUBDIR = "encoded"
SOURCE_HOST = os.environ.get("SOURCE_HOST", "gpu1")
