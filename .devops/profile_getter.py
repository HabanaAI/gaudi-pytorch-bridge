#!/usr/bin/env python3
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
        description="Script for retrieving information from json file describing build profiles"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--get-pt-requirement",
        action="store",
        nargs=1,
        metavar=("pt_version_id",),
        help="Prints required PyTorch pip package for pt_version_id.",
    )
    actions.add_argument("--get-version-literal", action="store", help="Prints version literal for version id provided")
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
    if args.get_version_literal:
        found = get_version_literal_and_source(args.get_version_literal)
        print(found.version if found else None)
    if args.get_cmakelists_supported_vers:
        print(get_cmakelists_supported_vers())
    if args.get_extras_version:
        print(get_extras_version(args.get_extras_version[0], args.get_extras_version[1]))
