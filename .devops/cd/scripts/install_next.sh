#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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

: ${1?"Usage: $0 major.minor.patch revision"}
: ${2?"Usage: $0 major.minor.patch revision"}

set -e

# Script params
# HABANA_RELEASE_VERSION - release version, e.g. '1.20.0'
# HABANA_RELEASE_ID - build id, e.g. '123'
HABANA_RELEASE_VERSION=$1
HABANA_RELEASE_ID=$2

# Env params
MIN_PYTHON_VER="${PYTHON_VERSION:-3}"
PIP_PYTHON_OPTIONS="${PYTHON_OPTIONS:-}"
EXTRA_INDEX_URL="${PYTHON_INDEX_URL:-}"
PYTHON_MPI_VERSION="${MPI_VERSION:-3.1.6}"

if [[ -z $SKIP_INSTALL_DEPENDENCIES ]]; then
  python${MIN_PYTHON_VER} -m pip install mpi4py=="${PYTHON_MPI_VERSION}" ${PIP_PYTHON_OPTIONS}
fi

if [[ -z $HABANALABS_LOCAL_DIR ]]; then
    python${MIN_PYTHON_VER} -m pip install habana-pyhlml=="${HABANA_RELEASE_VERSION}"."${HABANA_RELEASE_ID}" ${PIP_PYTHON_OPTIONS} ${EXTRA_INDEX_URL}
else
    python${MIN_PYTHON_VER} -m pip install ${HABANALABS_LOCAL_DIR}/habana_pyhlml-${HABANA_RELEASE_VERSION}.${HABANA_RELEASE_ID}*.whl ${PIP_PYTHON_OPTIONS} --disable-pip-version-check
fi
python${MIN_PYTHON_VER} -m pip install ./*.whl -r requirements-pytorch.txt ${PIP_PYTHON_OPTIONS} --disable-pip-version-check --no-warn-script-location

python${MIN_PYTHON_VER} -m pip uninstall -y pillow 2>/dev/null || echo "Skip uninstalling pillow. Need SUDO permissions."
python${MIN_PYTHON_VER} -m pip uninstall -y pillow-simd 2>/dev/null || echo "Skip uninstalling pillow-simd. Need SUDO permissions."
CC="cc -mavx2" python${MIN_PYTHON_VER} -m pip install -U --force-reinstall git+https://github.com/aostrowski-hbn/pillow-simd.git@simd/9.5.x ${PIP_PYTHON_OPTIONS} --disable-pip-version-check
