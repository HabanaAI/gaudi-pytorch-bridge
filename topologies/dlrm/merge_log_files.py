import torch
import numpy as np
import os
import glob
import argparse
import shutil

from distributed_utils import dlrm_get_emb_table_map

def combine_mchip_files(ip_path, world_size, ln_emb):
    for rank in range(world_size):
        rank_path = ip_path + '_' + str(rank)
        if rank == 0:
            if os.path.exists(ip_path):
                shutil.rmtree(ip_path)
            shutil.copytree(rank_path, ip_path)
            rank_files = glob.glob(rank_path + '/**/*.pt', recursive=True)
            for f in rank_files:
                op_file_path = os.path.join(ip_path, os.path.relpath(f, rank_path))
                dir_name, file_name = os.path.split(op_file_path)
                os.remove(op_file_path)
                if ('top_l.' in file_name or 'bot_l.' in file_name) and 'emb_l' not in file_name:
                    op_file_path = os.path.join(dir_name, file_name)
                    shutil.copyfile(f, op_file_path)

        rank_files = glob.glob(rank_path + '/**/*emb_l*.pt', recursive=True)
        rank_table, all2all_reorder = dlrm_get_emb_table_map(ln_emb, rank, world_size)
        for f in rank_files:
            dir_name, file_name = os.path.split(f)
            emb_l_name = file_name.split('.')
            emb_l_name[1] = str(rank_table[int(emb_l_name[1])])
            op_file_path = os.path.join(dir_name, '.'.join(emb_l_name))
            op_file_path = os.path.join(ip_path, os.path.relpath(op_file_path, rank_path))
            shutil.copyfile(f, op_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine multichip logs')
    parser.add_argument('--ip-path', type=str, help='Log file path')
    parser.add_argument('--world-size', type=int, default=2, help='World size')
    parser.add_argument('--arch-embedding-size', type=str, default="4-3-2")
    args = parser.parse_args()
    ln_emb = list(map(int, args.arch_embedding_size.split('-')))
    combine_mchip_files(args.ip_path, args.world_size, ln_emb)
