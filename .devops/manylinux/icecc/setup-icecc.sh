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
