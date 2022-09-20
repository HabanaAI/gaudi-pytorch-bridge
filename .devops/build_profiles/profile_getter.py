#!/usr/bin/env python3
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
import json
import os
import argparse
from enum import Enum


def _get_profiles_json():
    if hasattr(_get_profiles_json, "PROFILES_JSON"):
        return _get_profiles_json.PROFILES_JSON

    if not hasattr(_get_profiles_json, "JSON_PATH"):
        _get_profiles_json.JSON_PATH = os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/build_profiles/profiles.json")

    with open(_get_profiles_json.JSON_PATH, mode='r') as profiles_fp:
        _get_profiles_json.PROFILES_JSON = json.load(profiles_fp)

    return _get_profiles_json.PROFILES_JSON


def get_version_literal(version_name):
    profiles_json = _get_profiles_json()
    available_pt_versions = profiles_json["pt_versions"]
    try:
        return available_pt_versions[version_name]["version"]
    except KeyError:
        raise RuntimeError(f"pt_version \"{version_name}\" is not defined")


def get_version_args(profile):
    if "pt_versions" in profile:
        if "wheels" in profile:
            raise RuntimeError("Selected profile has both pt_versions and wheels attributes")
        selected_pt_versions = [get_version_literal(version) for version in profile["pt_versions"]
                                if get_version_literal(version) is not None]

        if not selected_pt_versions:
            raise RuntimeError("Selected profile does not specify any valid pt-versions to build")
        return ["--pt-versions", *selected_pt_versions]
    elif "wheels" in profile:
        wheel_spec = []
        all_wheels = _get_profiles_json()["wheels"]
        for wheel_id in profile["wheels"]:
            wheel = all_wheels[wheel_id]
            continue_on_error = "continue_on_error" in wheel and wheel["continue_on_error"]
            selected_pt_versions = [get_version_literal(version) for version in wheel["pt_versions"]
                                    if get_version_literal(version) is not None]
            if not selected_pt_versions:
                if continue_on_error:
                    continue
                raise RuntimeError(f"Wheel {wheel_id} in selected profile does not specify any valid pt-versions to build")
            continue_on_error_suffix = "optional" if continue_on_error else "standard"
            wheel_spec.append(f"{wheel['wheel_name']}:{','.join(selected_pt_versions)}:{continue_on_error_suffix}")

        if not wheel_spec:
            raise RuntimeError("Selected profile does not specify any valid wheels to build")

        return ["--build-whl", "--wheel-spec", *wheel_spec]

    raise RuntimeError("Selected profile has neither pt_versions nor wheels attribute")


def get_args_for_profile(profile_name):
    profiles_json = _get_profiles_json()
    selected_profile = profiles_json["profiles"][profile_name]
    additional_build_flags = selected_profile["additional_build_flags"] if "additional_build_flags" in selected_profile else []
    version_args = get_version_args(selected_profile)
    return additional_build_flags + version_args


def get_available_profiles():
    profiles_json = _get_profiles_json()
    return list(profiles_json["profiles"].keys())


def get_available_versions():
    profiles_json = _get_profiles_json()
    available_versions = [version_spec["version"] for version_spec in profiles_json["pt_versions"].values()
                          if version_spec["version"] is not None and version_spec["version"] != "nightly"]

    return available_versions


class RequirementPurpose(Enum):
    BUILD = "build"
    RUNTIME = "runtime"


def get_required_pt_package_name(pt_ver, purpose):
    profiles_json = _get_profiles_json()
    required_pt = profiles_json['required_pt']
    if pt_ver in required_pt:
        req = required_pt[pt_ver]
    else:
        req = required_pt['default']

    if isinstance(req, dict):
        return req[purpose.value]

    return req


def get_required_pt(pt_ver, purpose):
    pt_package_name = get_required_pt_package_name(pt_ver, purpose)
    if pt_ver == "nightly":
        return pt_package_name
    else:
        return f"{pt_package_name}=={pt_ver}"


def get_wheel_install_requires(pt_versions):
    required_pts = set([get_required_pt_package_name(pt_ver.label, RequirementPurpose.RUNTIME) for pt_ver in pt_versions])
    if len(required_pts) != 1:
        return ""

    return f"{required_pts.pop()} >= {min(pt_versions)}, <= {max(pt_versions)}"


def check_profile_file_integrity():
    from jsonschema import validate
    with open(os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/build_profiles/profiles.schema.json"), mode='r') as schema_fp:
        schema = json.load(schema_fp)
    validate(instance=_get_profiles_json(), schema=schema)

    for profile in get_available_profiles():
        _ = get_args_for_profile(profile)

    for ver in get_available_versions() + ["nightly"]:
        _ = get_required_pt(ver, RequirementPurpose.RUNTIME)
        _ = get_required_pt(ver, RequirementPurpose.BUILD)

    print("OK")


def get_cmakelists_supported_vers():
    return ";".join({f"{version[0]}\\.{version[1]}\\..*" for version in map(lambda ver: ver.split('.'), get_available_versions())})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for retrieving information from json file describing build profiles")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--get-pt-requirement",
        action="store",
        nargs=1,
        metavar=('pt_version_id',),
        help="Prints required PyTorch pip package for pt_version_id."
    )
    actions.add_argument("--get-version-literal", action="store", help="Prints version literal for version id provided")
    actions.add_argument("--get-cmakelists-supported-vers", action="store_true", help="Prints value that Torch_SUPPORTED_VERSIONS should be set to in CMakeLists")
    actions.add_argument("--check", action="store_true", help="Checks profile file integrity")
    parser.add_argument("--profiles", action="store", help="Allows providing of custom profile json")
    args = parser.parse_args()

    if args.profiles:
        _get_profiles_json.JSON_PATH = args.profiles
    if args.check:
        check_profile_file_integrity()
        exit()
    if args.get_pt_requirement:
        print(get_required_pt(args.get_pt_requirement[1], get_version_literal(args.get_pt_requirement[0]), RequirementPurpose.RUNTIME))
    if args.get_version_literal:
        print(get_version_literal(args.get_version_literal))
    if args.get_cmakelists_supported_vers:
        print(get_cmakelists_supported_vers())
