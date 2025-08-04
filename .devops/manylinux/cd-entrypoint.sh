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

set -e

RUN_ARGS=${@:-habana_help && echo -e '\\nPlease provide the command to run as arguments.'}

HABANA_ARTIFACTORY_SERVER="artifactory-kfs.habana-labs.com"
HABANA_PIP_INDEX="https://${HABANA_ARTIFACTORY_SERVER}/artifactory/api/pypi/pypi-virtual"
HABANA_PIP_INDEX_URL="$HABANA_PIP_INDEX/simple"
# shellcheck disable=SC2034  # the var is used when sourcing habana_env
SET_ABSOLUTE_HABANA_ENV=yes
WORK=${_WORK:-$HOME}
STACK=${_STACK:-$HOME/repos}
BUILD=${_BUILD:-$HOME/builds}
source $STACK/automation/habana_scripts/habana_env $WORK $STACK $BUILD

if [ -z "${HABANA_NO_VENV}" ]; then
    if ! test -f $WORK/.venv/bin/activate; then
        echo "Initializing Python $HABANA_PYTHON_VERSION virtual environment in $WORK/.venv"
        python${HABANA_PYTHON_VERSION} -m venv $WORK/.venv
        source $WORK/.venv/bin/activate
        cat << EOF > "$VIRTUAL_ENV"/pip.conf
[global]
index =  $HABANA_PIP_INDEX
index-url = $HABANA_PIP_INDEX_URL
trusted-host = $HABANA_ARTIFACTORY_SERVER
EOF
        pip${HABANA_PYTHON_VERSION} install --disable-pip-version-check --progress-bar=off -U pip
        pip${HABANA_PYTHON_VERSION} install --disable-pip-version-check --progress-bar=off -r $STACK/pytorch-integration/requirements.txt
    else
        echo "Activating existing Python $HABANA_PYTHON_VERSION virtual environment in $WORK/.venv"
        source $WORK/.venv/bin/activate
    fi
fi

eval $RUN_ARGS
exit $?
