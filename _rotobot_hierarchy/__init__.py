"""Vendored from rotobot-nuke: the v3 reader and the camera / person / part
decomposition the Silhouette converter builds its layer hierarchy from.

A private name on purpose: the full ``rotobot_nuke`` package (with the Nuke
importer and undersampler) may also be installed, and this partial copy must
never shadow it. ``reader.py`` and ``hierarchy.py`` are byte-for-byte copies
of the commit in VENDORED.txt; packaging/ci/check-vendored-hierarchy.sh fails
CI if they drift. Change them upstream, then re-vendor.
"""
