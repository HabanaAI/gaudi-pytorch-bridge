#!/bin/bash
###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

# "$COMMON_GIT_ARGS" break git clone, it have to be unquoted.
# To avoid clutter it's disabled for entire file.
# shellcheck disable=SC2086

set -e
set -x

# This script accept one argument with localization where dependencies have to be copied.
if [ $# != 1 ]; then
  echo "Only one argument with target directory is accepted."
fi

# try to create target directory
mkdir -p "$1"
pushd "$1"

COMMON_GIT_ARGS="-c advice.detachedHead=false --quiet --depth=1 --single-branch --shallow-submodules --recurse-submodules --jobs=$(nproc)"

# ClangTidy:
git clone $COMMON_GIT_ARGS https://github.com/matus-chochlik/ctcache.git
pushd ctcache
git checkout b54f74807fc02c8897247fda6229aabbac78a560
popd

# Core Deps
git clone $COMMON_GIT_ARGS --branch 0.0.3-cmake https://github.com/ArashPartow/exprtk.git
git clone $COMMON_GIT_ARGS --branch v0.8.3 https://github.com/Cyan4973/xxHash.git xxhash
git clone $COMMON_GIT_ARGS --branch v3.12.0 https://github.com/nlohmann/json.git nlohmann_json
git clone $COMMON_GIT_ARGS --branch 9.1.0 https://github.com/fmtlib/fmt.git
git clone $COMMON_GIT_ARGS --branch v0.9.7 https://github.com/Neargye/magic_enum.git
wget --no-verbose https://snapshot.debian.org/archive/debian/20250412T205410Z/pool/main/d/devscripts/devscripts_2.25.9.tar.xz -O devscripts.tar.xz
tar xf devscripts.tar.xz

# Python helpers
git clone $COMMON_GIT_ARGS --branch 20250512.1 https://github.com/abseil/abseil-cpp.git

# SLRG
git clone $COMMON_GIT_ARGS --branch v2.4.12 https://github.com/doctest/doctest.git

# tests
git clone $COMMON_GIT_ARGS --branch v1.17.0 https://github.com/google/googletest.git
