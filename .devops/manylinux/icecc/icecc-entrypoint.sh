#!/usr/bin/env bash
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

# If iceccd can't connect to a scheduler, make sure you're docker running with --net=host.
# For printing full debug logs to console, replace `-d` with `-vvv &`
iceccd --nice 10 -u icecc -b /var/cache/icecc --no-remote -p 10246 -N "${HOSTNAME}"-manylinux -d

# shellcheck source=../entrypoint.sh
exec entrypoint.sh "$@"
