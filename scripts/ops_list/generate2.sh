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

PDIR=`python get_python_libs_dir.py`
echo PDIR=$PDIR
echo HABANA_NPU_STACK_PATH=$HABANA_NPU_STACK_PATH

rm -f ops_table.csv ops_table_filtered.csv ops_table_by_prio.csv
python ops_summary.py -c ops.cpu -g ops.gpu -hh ops.hpu -p $PDIR -n $HABANA_NPU_STACK_PATH > ops_table.csv
python ops_summary.py -c ops.cpu -g ops.gpu -hh ops.hpu -p $PDIR -n $HABANA_NPU_STACK_PATH -f > ops_table_filtered.csv
python ops_summary.py -c ops.cpu -g ops.gpu -hh ops.hpu -p $PDIR -n $HABANA_NPU_STACK_PATH -prio > ops_table_by_prio.csv

# To generate kr_ops.hpu use this:
# https://gerrit.habana-labs.com/#/c/327362/
# Run any simple test and filter stdout

rm -f aten_ops_table.csv
python aten_ops_summary.py -hh kr_ops.hpu > aten_ops_table.csv
