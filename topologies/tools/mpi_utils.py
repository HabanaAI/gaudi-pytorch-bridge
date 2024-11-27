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

import socket
from mpi4py import MPI

def get_root_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        root_ip = ip_address
    else:
        root_ip = None

    root_ip = comm.bcast(root_ip,root=0)
    return (root_ip)


def get_my_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank

def get_hcl_config(pernode):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    ip_list = comm.allgather(ip_address)
    str = "{\"HCL_PORT\": 5332,\"HCL_TYPE\": \"HLS1\",\"HCL_RANKS\": ["

    for ip in ip_list:
        for i in range(0,pernode):
            str = str + "\"" + ip + "\","
    str = str[:-1]
    str = str + "]}"
    return str
