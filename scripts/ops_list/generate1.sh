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

TVER=(`python get_torch_version.py`)

tver=${TVER[0]}
opt=${TVER[1]}
f=ops.${TVER[2]}

echo TVER=$TVER
echo $TVER >> torch.__version__

rm -f $f
python ops_list.py -dk $opt > $f
python ops_list.py -dk $opt -a >> $f

