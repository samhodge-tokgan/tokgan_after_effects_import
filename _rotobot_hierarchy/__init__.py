"""Vendored from rotobot-nuke: the v3 reader, the camera / person / part
decomposition the converters build their layer hierarchy from, and the RDP
keyframe undersampler the queue offers as an export option.

A private name on purpose: the full ``rotobot_nuke`` package (with the Nuke
importer) may also be installed, and this partial copy must never shadow it.
That invariant is why the undersampler is reachable here as
``_rotobot_hierarchy.undersample`` and never as ``rotobot_nuke.undersample``,
even though the file is byte-identical to it.

``reader.py``, ``hierarchy.py`` and ``undersample.py`` are byte-for-byte
copies of the commit in VENDORED.txt, whose ``files`` line is the single list
of what is vendored -- packaging/ci/check-vendored-hierarchy.sh reads that
line and fails CI if any copy drifts. Change them upstream, then re-vendor
with ``--update <commit>``; editing them here is reverted by the next run,
and ``git checkout`` will not restore them because they are not tracked
individually.
"""
