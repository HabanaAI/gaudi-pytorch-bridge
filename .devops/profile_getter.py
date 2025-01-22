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

import argparse
import sys

from build_profiles.profiles import (
    RequirementPurpose,
    check_profile_file_integrity,
    get_cmakelists_supported_vers,
    get_extras_version,
    get_profiles_json,
    get_required_pt,
    get_version_literal_and_source,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for retrieving information from JSON file describing build profiles"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--get-pt-requirement",
        metavar=("pt_version_id",),
        help="Prints required PyTorch pip package for given pt_version_id (e.g. current). Skips the patch version.",
    )
    actions.add_argument(
        "--get-strict-pt-requirement",
        metavar=("pt_version_id",),
        help="Prints required PyTorch pip package for given pt_version_id (e.g. current), including the patch version.",
    )
    actions.add_argument("--get-version-literal", action="store", help="Prints version literal for provided version ID")
    actions.add_argument(
        "--get-cmakelists-supported-vers",
        action="store_true",
        help="Prints value that Torch_SUPPORTED_VERSIONS should be set to in CMakeLists",
    )
    actions.add_argument("--check", action="store_true", help="Checks profile file integrity")
    actions.add_argument(
        "--get-extras-version",
        action="store",
        nargs=2,
        metavar=("package_name", "pt_version_id"),
        help="Prints version of requested extra package (e.g. torchaudio) for given pt_version_id (e.g. current)",
    )
    parser.add_argument("--profiles", action="store", help="Allows providing of custom profile json")
    args = parser.parse_args()

    if args.profiles:
        get_profiles_json.JSON_PATH = args.profiles
    if args.check:
        check_profile_file_integrity()
        sys.exit()
    if args.get_pt_requirement:
        print(
            get_required_pt(get_version_literal_and_source(args.get_pt_requirement).version, RequirementPurpose.RUNTIME)
        )
    if args.get_strict_pt_requirement:
        print(
            get_required_pt(
                get_version_literal_and_source(args.get_strict_pt_requirement, strict=True).version,
                RequirementPurpose.RUNTIME,
            )
        )
    if args.get_version_literal:
        found = get_version_literal_and_source(args.get_version_literal)
        print(found.version if found else None)
    if args.get_cmakelists_supported_vers:
        print(get_cmakelists_supported_vers())
    if args.get_extras_version:
        print(get_extras_version(args.get_extras_version[0], args.get_extras_version[1]))
