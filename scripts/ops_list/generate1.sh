#!/bin/bash
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

TVER=(`python get_torch_version.py`)

tver=${TVER[0]}
opt=${TVER[1]}
f=ops.${TVER[2]}

echo TVER=$TVER
echo $TVER >> torch.__version__

rm -f $f
python ops_list.py -dk $opt > $f
python ops_list.py -dk $opt -a >> $f

