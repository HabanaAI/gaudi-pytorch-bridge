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
# Uploads images built using build.sh to Artifactory
#
regular_image=artifactory-kfs.habana-labs.com/docker/manylinux/al2
icecc_image=artifactory-kfs.habana-labs.com/developers-docker-dev-local/manylinux/al2-with-icecc

date_tag=$(date +%Y%m%d_%H%M%S)

docker tag $regular_image:latest $regular_image:"$date_tag"
docker tag $icecc_image:latest $icecc_image:"$date_tag"

docker push $regular_image:latest
docker push $regular_image:"$date_tag"

docker push $icecc_image:latest
docker push $icecc_image:"$date_tag"

