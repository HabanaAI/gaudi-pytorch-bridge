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
#

# Dockerfile for base image: https://github.com/pytorch/pytorch/blob/main/.ci/docker/manywheel/Dockerfile_2_28
# Latest tag can be found at: https://hub.docker.com/r/pytorch/manylinux2_28-builder/tags?name=cpu
FROM pytorch/manylinux2_28-builder:cpu-87cefcaa420ab3b6d6e06e63aa33970b3723d060

# Provide all versions of Python and Pip in PATH
RUN for py in /opt/python/cp*[0-9]/bin/{python3,pip3}.*; do \
        ln -sf $py /usr/bin/$(basename $py); \
    done

# Here's why we install specific packages below:
# * numactl-libs (libnuma.so.1) are required to import pytorch-fork
# * zlib is required to build integration
# * sqlite, sox, and curl are required to build aeon which we depend on
# * protobuf-compiler is required to for graph_visualization in dynamo debug utils
# * redhat-lsb-core provides lsb_release required by automation scripts, e.g. for building aeon
RUN dnf update -y && dnf install -y \
        ccache \
        sudo \
        passwd \
        numactl-libs \
        zlib-devel \
        sqlite-devel \
        sox-devel \
        curl-devel \
        protobuf-compiler \
        redhat-lsb-core \
    && dnf clean all

COPY --chmod=755 entrypoint.sh cd-entrypoint.sh /usr/local/bin/
ENTRYPOINT [ "entrypoint.sh" ]
