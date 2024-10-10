#!/usr/bin/env python
###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any, Optional, Type, Union

import packaging.version
import requests
from pkginfo import Wheel

log = logging.getLogger(__file__)


class Version(packaging.version.Version):
    """A PEP-440-compliant version with extensions for version matching, labels,
    comparisons, etc.

    PyPA reference for PEP-440: https://packaging.pypa.io/en/stable/version.html
    """

    def __init__(self, version: Union[str, Type[sys.version_info]], label: Optional[str] = None) -> None:
        """
        :param version a string in PEP-440 format or a Python system version
        :param label   optional custom symbolic label of the version. Used to generate @see Version.label.
        """
        self._label = label
        self.wheel_path = None
        self.wheel_url = None
        if isinstance(version, str):
            if version.startswith("file://"):
                version = self._handle_wheel(version.split("file://")[-1])
            if version.startswith("https://"):
                version = self._handle_url(version)
            super().__init__(version)
        elif isinstance(version, type(sys.version_info)):
            version = f"{version.major}.{version.minor}.{version.micro}"
            super().__init__(version)
        else:
            raise TypeError(f"Version must be a string or a sys.version_info: {version}")

    def __eq__(self, other: Any) -> bool:
        try:
            rhs = Version(other) if isinstance(other, str) else other
            return super().__eq__(rhs)
        except packaging.version.InvalidVersion:
            return False

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if self.wheel_url:
            return f"<Version('{self}', source={self.wheel_url})>"
        if self.wheel_path:
            return f"<Version('{self}', source=file://{self.wheel_path})>"
        return super().__repr__()

    def _handle_wheel(self, path):
        self.wheel_path = path
        whl = Wheel(path)
        return whl.version

    def _handle_url(self, url):
        self.wheel_url = url

        with tempfile.TemporaryDirectory() as tmp_dir:
            log.info(f"Trying to download wheel from {url} to {tmp_dir.name}")

            req = requests.get(url, stream=True, timeout=300)  # timeout is in secs

            if not req.ok:
                raise ConnectionError(f"Download failed: status code {req.status_code}\n{req.text}")

            filename = url.split("/")[-1]
            wheel_path = os.path.join(tmp_dir.name, filename)
            with open(wheel_path, "wb") as downloaded_wheel:
                for chunk in req.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        downloaded_wheel.write(chunk)
                        downloaded_wheel.flush()
                        os.fsync(downloaded_wheel.fileno())
            return self._handle_wheel(wheel_path)

    @property
    def label(self):
        """
        Either a customized version name like "nightly" or version string built
        out of version numbers. For usage @see prepare_venv in build.py
        """
        return self._label if self._label else str(self)

    def significant_matches(self, candidate: Version) -> bool:
        """Checks if all version components of wildcard (i.e. self) and the
        compute platform match those from candidate.

        For instance:
        Version(" ").significant_matches(candidate=Version("2.2.3.4a0")) == True
        Version("1.3+cpu").significant_matches(Version("1.3.0+cpu")) == True

        Assume git hashes in local part of version mean a CPU platform.
        """
        wildcard, candidate_ver = self.release, candidate.release
        assert len(wildcard) <= len(candidate_ver)
        release_matches = all(w == c for w, c in zip(wildcard, candidate_ver))

        if not self.is_prerelease:
            return release_matches
        return release_matches and self.pre == candidate.pre


def is_official_stable_cpu_version(pt_ver: Version) -> bool:
    return not pt_ver.is_prerelease and pt_ver.local == "cpu" and not pt_ver.is_devrelease


def is_official_nightly_cpu_version(pt_ver: Version) -> bool:
    return not pt_ver.is_prerelease and pt_ver.local == "cpu" and len(str(pt_ver.dev)) == len("20190731")


def is_wheel_version(pt_ver: Version) -> bool:
    return pt_ver.wheel_path
