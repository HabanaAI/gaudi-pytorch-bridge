#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
