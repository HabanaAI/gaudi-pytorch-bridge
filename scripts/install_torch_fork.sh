#!/usr/bin/env bash
###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

# Downloads and installs the PyTorch-fork wheel from Vault.
# Resolves the version to download based on the current (main) build profile.
# Args:
# - $1 - NPU stack version, e.g. 1.23.4
# - $2 - build number, e.g. 567

set -e
set -u

main() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 <version> <build>"
        exit 1
    fi

    local version build
    version=$1
    shift
    build=$1
    shift
    readonly version build

    local current_torch
    current_torch=$("$PYTORCH_MODULES_ROOT_PATH"/.devops/profile_getter.py --get-strict-pt-requirement current)
    current_torch="${current_torch//*==/}"
    readonly current_torch

    local storage
    storage=$(mktemp -d 2>/dev/null || mktemp -d -t 'torch-fork-wheels')
    readonly storage

    source /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        if [[ "$VERSION_ID" == "22.04" ]]; then
            os="ubuntu2204"
        elif [[ "$VERSION_ID" == "24.04" ]]; then
            os="ubuntu2404"
        fi
    elif [[ "$ID" == "rhel" ]]; then
        if [[ "$VERSION_ID" == "86" ]]; then
            os="rhel86"
        elif [[ "$VERSION_ID" == "92" ]]; then
            os="rhel92"
        elif [[ "$VERSION_ID" == "94" ]]; then
            os="rhel94"
        fi
    elif [[ "$ID" == "suse" ]]; then
        os="suse155"
    elif [[ "$ID" == "tencentos" ]]; then
        os="tencentos31"
    fi

    # shellcheck disable=SC2064  # Otherwise storage is already out of scope
    trap "rm -r \"$storage\"" EXIT

    wget "https://vault.habana.ai/artifactory/gaudi-pt-modules/${version}/${build}/pytorch/${os}/pytorch_modules-v${current_torch}_${version}_${build}.tgz" -O - \
        | tar -C "$storage" -xz --wildcards --no-anchored "torch-${current_torch}*"
    pip install "$storage/torch-${current_torch}"*
}

main "$@"
