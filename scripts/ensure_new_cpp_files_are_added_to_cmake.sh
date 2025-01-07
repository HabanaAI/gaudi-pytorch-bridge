#!/usr/bin/env bash
###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

#
# Checks if any CMakeLists.txt file contains the .pp files that were added,
# copied, pr renamed in a Git diff.
# Accepts filenames to check as arguments. It will filter .cpp files from them.
#

# Filters filenames by printing those ending in .cpp
# $@ - filenames
print_cpp_files() {
    for file in "$@"; do
        if [[ "$file" =~ .cpp$ ]]; then
            echo "$file"
        fi
    done
}

# Given a file name and a list of files returns:
# - 0 -- if the first file is present in any of the files from the file list
# - 1 -- otherwise
#
# $1 - file to search for
# $2+ - files to grep in
is_file_in_lists() {
    needle=$1
    shift
    # echo "$@" | xargs grep -q "$needle"
    for LIST in "$@"; do
        PREFIX="$(dirname "$LIST")"  # e.g. .../scripts/../tests/
        PREFIX="${PREFIX//$WORKDIR\//}"  # e.g. tests/
        grep -q "${needle//$PREFIX\//}" "$LIST" && return 0
    done
    return 1
}

CPP_FILENAMES=$(print_cpp_files "$@")

WORKDIR="$(dirname "$(realpath "$0")")/.." # assume we're in project_dir/scripts
mapfile -t CMAKELISTS < <(find "$WORKDIR" -name CMakeLists.txt)

RET=0
for file in $CPP_FILENAMES; do
    if ! is_file_in_lists "$file" "${CMAKELISTS[@]}"; then
         echo "Source file $file is not present in any CMakeLists.txt file. Please add it so it's compiled during the build"
         RET=$((RET + 1))
    fi
done

exit $RET
