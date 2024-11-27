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
# Builds manylinux images: with and without icecc support
# To upload those images to Artifactory use upload-to-rt.sh
#

set -e

echo Building AL2-based Manylinux docker
docker build -t artifactory-kfs.habana-labs.com/docker/manylinux/al2:latest .

echo
echo Building an icecc-enabled Manylinux docker based on the above one
docker build -t artifactory-kfs.habana-labs.com/developers-docker-dev-local/manylinux/al2-with-icecc:latest icecc
