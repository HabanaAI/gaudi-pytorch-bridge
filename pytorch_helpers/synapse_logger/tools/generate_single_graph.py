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


import glob
import os.path
import re
import sys
from os import fdopen, remove
from shutil import copymode, move
from tempfile import mkstemp


def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh, "w") as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)


def prepend_line(file_name, line):
    dummy_file = file_name + ".bak"
    with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
        write_obj.write(line + "\n")
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)


def add_endofgraph(file_name):
    f = open(file_name, "a")
    f.write("}")
    f.close()


## main
os.system(
    "rm -rf single_graph.pdf; python pytorch_helpers/synapse_logger/tools/browse_log.py draw ;  rm -rf .graph_dumps/*.svg; rm -rf .graph_dumps/*.used"
)

single_graph_file_name = "single_graph.dot"
list_of_files = glob.glob(".graph_dumps/*")

for file_name in list_of_files:
    f = open(file_name, "r")
    lst = []
    for line in f:
        lst.append(line)
    lst.append("\n")
    f.close()

    f = open(single_graph_file_name, "a")
    for line in lst:
        f.write(line)
    f.close()

replace(single_graph_file_name, "digraph", "subgraph")
prepend_line(single_graph_file_name, "digraph single_graph{")
add_endofgraph(single_graph_file_name)

os.system("dot -Tpdf single_graph.dot -o single_graph.pdf ; rm -f single_graph.dot")
