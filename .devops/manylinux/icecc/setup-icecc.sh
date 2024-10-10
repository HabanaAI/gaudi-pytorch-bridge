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
# Downloads, builds, and installs Icecream.
#
# Uses yum to install build dependencies.
#

set -e

yum install -y libcap-ng-devel libarchive-devel lzo-devel libzstd-devel

mkdir -p /tmp/icecc
pushd /tmp/icecc || exit 1

wget 'https://github.com/icecc/icecream/releases/download/1.4/icecc-1.4.0.tar.xz'
xz -d icecc-1.4.0.tar.xz
tar -xf icecc-1.4.0.tar
rm icecc-1.4.0.tar

pushd icecc-1.4.0 || exit 1

./configure --prefix=/opt/icecream
make -j "$(nproc)"
make install
# binaries will be in /opt/icecream/{bin,sbin} - you might want to add them to PATH

popd 2 || exit 1
rm -r /tmp/icecc

adduser --create-home --home-dir /var/cache/icecc --shell /bin/false --system --user-group icecc
