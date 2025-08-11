#!/usr/bin/env bash
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

glibc_version=$(getconf GNU_LIBC_VERSION | tr ' ' '\n' | tail -n 1)
echo Used version of GLIBC: $glibc_version

libhl_logger_path=${BUILD_ROOT_LATEST}/libhl_logger.so
# Check if the libhl_logger.so exists
if [ -f "$libhl_logger_path" ]; then
    # Check if the libhl_logger.so is compatible with the installed GLIBC
    printf "%s\n%s\n" "$(objdump -T "$libhl_logger_path" | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -uV)" "$glibc_version" | sort -VC
    result=$?
    if [ $result -ne 0 ]; then
        echo "Available libhl_logger.so is incompatible with the installed version of GLIBC. It will be rebuild now."
    else
        echo "Available libhl_logger.so is compatible with the installed version of GLIBC. No need to rebuild hl_logger."
        exit 0
    fi
else
    echo "Libhl_logger.so doesn't exist. It will be rebuild now."
fi
echo "Building hl_logger with the command: build_hl_logger "$@""
build_hl_logger "$@"
exit $?
