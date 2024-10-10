#!/bin/bash
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
