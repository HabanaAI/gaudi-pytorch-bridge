#!/usr/bin/env python
###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
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
import sys
from typing import Any, Optional, Type, Union
from pkginfo import Wheel

import packaging.version

import tempfile
import logging
import requests
import os

log = logging.getLogger(__file__)


class Version(packaging.version.Version):
    """A PEP-440-compliant version with extensions for version matching, labels,
    comparisons, etc.

    PyPA reference for PEP-440: https://packaging.pypa.io/en/stable/version.html
    """
    tmp_dir = tempfile.TemporaryDirectory()

    def __init__(
        self, version: Union[str, Type[sys.version_info]], label: Optional[str] = None
    ) -> None:
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
            raise TypeError(
                f"Version must be a string or a sys.version_info: {version}"
            )

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
        elif self.wheel_path:
            return f"<Version('{self}', source=file://{self.wheel_path})>"
        else:
            return super().__repr__()

    def _handle_wheel(self, path):
        self.wheel_path = path
        w = Wheel(path)
        return w.version

    def _handle_url(self, url):
        self.wheel_url = url
        log.info(f"Trying to download wheel from {url} to {Version.tmp_dir.name}")
        r = requests.get(url, stream=True)
        if r.ok:
            filename = url.split('/')[-1]
            wheel_path = os.path.join(Version.tmp_dir.name, filename)
            with open(wheel_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
            return self._handle_wheel(wheel_path)
        else:
            raise ConnectionError("Download failed: status code {}\n{}".format(r.status_code, r.text))

    @property
    def label(self):
        """
        Either a customized version name like "nightly" or version string built
        out of version numbers. For usage @see prepare_venv in build.py
        """
        return self._label if self._label else str(self)

    def _platform_matches(self, candidate: Version) -> bool:
        def get_platform(plat: Optional[str]):
            if not plat or plat.startswith("git"):
                return "cpu"
            return plat

        lhs_platform = get_platform(self.local)
        rhs_platform = get_platform(candidate.local)
        return lhs_platform == rhs_platform

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

        if not self._platform_matches(candidate):
            return False

        if not self.is_prerelease:
            return release_matches
        else:
            return release_matches and self.pre == candidate.pre


def is_official_stable_cpu_version(pt_ver: Version) -> bool:
    return (
        not pt_ver.is_prerelease and pt_ver.local == "cpu" and not pt_ver.is_devrelease
    )


def is_official_nightly_cpu_version(pt_ver: Version) -> bool:
    return (
        not pt_ver.is_prerelease
        and pt_ver.local == "cpu"
        and len(str(pt_ver.dev)) == len("20190731")
    )


def is_pt_fork_version(pt_ver: Version) -> bool:
    return pt_ver.pre == ("a", 0) and pt_ver.local.startswith("git")


def is_wheel_version(pt_ver: Version) -> bool:
    return pt_ver.wheel_path
