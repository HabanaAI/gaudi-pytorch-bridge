#!/usr/bin/env bash
: ${1?"Usage: $0 major.minor.patch revision"}
: ${2?"Usage: $0 major.minor.patch revision"}
set -e
# note
# script params
HABANA_RELEASE_VERSION=$1
HABANA_RELEASE_ID=$2
# env params
MIN_PYTHON_VER="${PYTHON_VERSION:-3}"
PIP_PYTHON_OPTIONS="${PYTHON_OPTIONS:-}"
EXTRA_INDEX_URL="${PYTHON_INDEX_URL:-}"
PYTHON_MPI_VERSION="${MPI_VERSION:-3.1.6}"
# define constants
PILLOW_SIMD_VERSION="9.5.0.post1"

if [ -z $SKIP_INSTALL_DEPENDENCIES ]; then
  python${MIN_PYTHON_VER} -m pip install mpi4py=="${PYTHON_MPI_VERSION}" ${PIP_PYTHON_OPTIONS}
fi

if [ -z $HABANALABS_LOCAL_DIR ]; then
    python${MIN_PYTHON_VER} -m pip install habana-pyhlml=="${HABANA_RELEASE_VERSION}"."${HABANA_RELEASE_ID}" ${PIP_PYTHON_OPTIONS} ${EXTRA_INDEX_URL}
else
    python${MIN_PYTHON_VER} -m pip install ${HABANALABS_LOCAL_DIR}/habana_pyhlml-${HABANA_RELEASE_VERSION}.${HABANA_RELEASE_ID}*.whl ${PIP_PYTHON_OPTIONS} --disable-pip-version-check
fi
python${MIN_PYTHON_VER} -m pip install ./*.whl -r requirements-pytorch.txt ${PIP_PYTHON_OPTIONS} --disable-pip-version-check --no-warn-script-location

python${MIN_PYTHON_VER} -m pip uninstall -y pillow 2>/dev/null || echo "Skip uninstalling pillow. Need SUDO permissions."
python${MIN_PYTHON_VER} -m pip uninstall -y pillow-simd 2>/dev/null || echo "Skip uninstalling pillow-simd. Need SUDO permissions."
python${MIN_PYTHON_VER} -m pip install pillow-simd==${PILLOW_SIMD_VERSION} ${PIP_PYTHON_OPTIONS} --disable-pip-version-check
