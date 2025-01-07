#!/usr/bin/env python3
###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

from __future__ import annotations

import json
import os
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Optional, Sequence


@dataclass(frozen=True)
class VersionLiteralAndSource:
    version: str
    source: str

    # allow unpacking
    def __iter__(self):
        return iter(astuple(self))

    def __lt__(self, other: VersionLiteralAndSource):
        # Deliberately moved here so that external pip packages are not required to just read the versions
        from .version import Version

        return Version(self.version) < Version(other.version)


def get_profiles_json():
    """Returns a JSON object with the contents of the profiles.json file"""
    if hasattr(get_profiles_json, "PROFILES_JSON"):
        return getattr(get_profiles_json, "PROFILES_JSON")

    if not hasattr(get_profiles_json, "JSON_PATH"):
        setattr(
            get_profiles_json,
            "JSON_PATH",
            os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/build_profiles/profiles.json"),
        )

    with open(getattr(get_profiles_json, "JSON_PATH"), mode="r", encoding="utf-8") as profiles_fp:
        setattr(get_profiles_json, "PROFILES_JSON", json.load(profiles_fp))

    return getattr(get_profiles_json, "PROFILES_JSON")


def get_version_literal_and_source(version_name: str, strict: bool = False) -> Optional[VersionLiteralAndSource]:
    profiles_json = get_profiles_json()
    available_pt_versions = profiles_json["pt_versions"]
    try:
        node = available_pt_versions[version_name]
        if node["version"] is None:
            return None

        if strict:
            version = node["version"]
        else:
            # just the major.minor version
            version = ".".join(node["version"].split(".", 2)[:2])

        return VersionLiteralAndSource(version, node["default_source"])
    except KeyError as exc:
        raise KeyError(f'pt_version "{version_name}" is not defined') from exc


def get_version_args(profile):
    if "pt_versions" in profile:
        if "wheels" in profile:
            raise RuntimeError("Selected profile has both pt_versions and wheels attributes")
        selected_pt_versions = [
            get_version_literal_and_source(version).version
            for version in profile["pt_versions"]
            if get_version_literal_and_source(version) is not None
        ]

        if not selected_pt_versions:
            raise RuntimeError("Selected profile does not specify any valid pt-versions to build")
        return ["--pt-versions", *selected_pt_versions]
    if "wheels" in profile:
        wheel_spec = []
        all_wheels = get_profiles_json()["wheels"]
        for wheel_id in profile["wheels"]:
            wheel = all_wheels[wheel_id]
            continue_on_error = "continue_on_error" in wheel and wheel["continue_on_error"]
            selected_pt_versions = [
                get_version_literal_and_source(version).version
                for version in wheel["pt_versions"]
                if get_version_literal_and_source(version) is not None
            ]
            if not selected_pt_versions:
                if continue_on_error:
                    continue
                raise RuntimeError(
                    f"Wheel {wheel_id} in selected profile does not specify any valid pt-versions to build"
                )
            continue_on_error_suffix = "optional" if continue_on_error else "standard"
            wheel_spec.append(f"{wheel['wheel_name']}:{','.join(selected_pt_versions)}:{continue_on_error_suffix}")

        if not wheel_spec:
            raise RuntimeError("Selected profile does not specify any valid wheels to build")

        return ["--build-whl", "--wheel-spec", *wheel_spec]

    raise RuntimeError("Selected profile has neither pt_versions nor wheels attribute")


def get_args_for_profile(profile_name):
    profiles_json = get_profiles_json()
    selected_profile = profiles_json["profiles"][profile_name]
    additional_build_flags = (
        selected_profile["additional_build_flags"] if "additional_build_flags" in selected_profile else []
    )
    version_args = get_version_args(selected_profile)
    return additional_build_flags + version_args


def get_available_profiles():
    profiles_json = get_profiles_json()
    return list(profiles_json["profiles"].keys())


def get_available_versions() -> Sequence[VersionLiteralAndSource]:
    profiles_json = get_profiles_json()
    available_versions = [
        VersionLiteralAndSource(version_spec["version"], version_spec["default_source"])
        for version_spec in profiles_json["pt_versions"].values()
        if version_spec["version"] is not None and version_spec["version"] != "nightly"
    ]

    return available_versions


class RequirementPurpose(Enum):
    BUILD = "build"
    RUNTIME = "runtime"


def get_required_pt_package_name(pt_ver, purpose) -> str:
    profiles_json = get_profiles_json()
    required_pt = profiles_json["required_pt"]
    if pt_ver in required_pt:
        req = required_pt[pt_ver]
    else:
        req = required_pt["default"]

    if isinstance(req, dict):
        return req[purpose.value]

    return req


def get_required_pt(pt_ver, purpose) -> str:
    pt_package_name = get_required_pt_package_name(pt_ver, purpose)
    if pt_ver == "nightly":
        return pt_package_name
    return f"{pt_package_name}=={pt_ver}"


def get_wheel_install_requires(pt_versions):
    required_pts = {get_required_pt_package_name(pt_ver.label, RequirementPurpose.RUNTIME) for pt_ver in pt_versions}
    if len(required_pts) != 1:
        return ""

    return f"{required_pts.pop()} >= {min(pt_versions)}, <= {max(pt_versions)}"


def check_profile_file_integrity():
    from jsonschema import validate  # pylint: disable=import-outside-toplevel

    with open(
        os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/build_profiles/profiles.schema.json"),
        mode="r",
        encoding="utf-8",
    ) as schema_fp:
        schema = json.load(schema_fp)
    validate(instance=get_profiles_json(), schema=schema)

    for profile in get_available_profiles():
        _ = get_args_for_profile(profile)

    for ver in get_available_versions() + ["nightly"]:
        _ = get_required_pt(ver, RequirementPurpose.RUNTIME)
        _ = get_required_pt(ver, RequirementPurpose.BUILD)

    print("OK")


def get_cmakelists_supported_vers():
    return ";".join(
        {
            f"{version[0]}\\.{version[1]}\\..*"
            for version in map(lambda ver_source: ver_source.version.split("."), get_available_versions())
        }
    )


def get_extras_version(package_name: str, pt_version_id: str) -> str:
    profiles_json = get_profiles_json()
    available_pt_versions = profiles_json["pt_versions"]
    try:
        node = available_pt_versions[pt_version_id]
        return node["extras"][package_name]
    except KeyError as exc:
        raise KeyError(f'{package_name} version for "{pt_version_id}" is not defined') from exc
