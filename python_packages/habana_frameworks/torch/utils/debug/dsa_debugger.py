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

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time

import numpy as np
import tqdm


def remove_file(path, verbose=True, strict=True):
    if os.path.isfile(path):
        if verbose:
            print(f"[INFO] Deleting file {path}")
        os.remove(path)
    else:
        if strict:
            raise ValueError(f"{path} is not a file")


def remove_dir(path, strict=True):
    if os.path.isdir(path):
        print(f"[INFO] Deleting directory {path}")
        shutil.rmtree(path)
    else:
        if strict:
            raise ValueError(f"{path} is not a directory")


def calc_difference(a, b):
    diff = np.subtract(a, b)
    abs_diff = np.abs(diff)
    abs_max = np.amax(abs_diff)
    abs_min = np.amin(abs_diff)

    mse = np.mean(np.square(diff))
    rmse = np.sqrt(mse)
    return {"abs_max": abs_max.item(), "abs_min": abs_min.item(), "mse": mse.item(), "rmse": rmse.item()}


def calc_similarity(
    a,
    b,
    threshold=0.0,
):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    norm_relative = np.divide(norm_a, norm_b)
    angle = np.arccos(min(np.dot(a, b) / norm_a / norm_b, 1.0)) / np.pi * 180
    angle = np.around(angle, 2)
    cosine_similarity = np.less_equal(angle, threshold)
    all_close = np.allclose(a, b)
    return {
        "norm_a": norm_a.item(),
        "norm_b": norm_b.item(),
        "norm_relative": norm_relative.item(),
        "angle": angle.item(),
        "cosine_similarity": cosine_similarity,
        "all_close": all_close,
    }


class DivergenceAnalyzer:
    def __init__(self, cfg, use_cache=True):
        self.cfg = cfg
        self.dumpdir = os.path.join(args.out)
        # For master Slave mode this is used as tmp dump location
        self.hls_local_dir = "/tmp/dumps_hls"
        self.logdir = os.path.join(self.dumpdir, "divergence_logs")
        self.dumpdir_static = os.path.join(self.dumpdir, "StaticSynRec")
        self.dumpdir_dynamic = os.path.join(self.dumpdir, "DynamicSynRec")
        self.hls_dumpdir_static = os.path.join(self.hls_local_dir, "StaticSynRec")
        self.hls_dumpdir_dynamic = os.path.join(self.hls_local_dir, "DynamicSynRec")
        self.dict_cache = None
        self.mismatch_map = None
        self.mismatch_static = None
        self.mismatch_dynamic = None
        self.use_cache = use_cache

        if not self.is_master_slave_config():
            if self.cfg.cmd is not None:
                remove_dir(self.dumpdir, strict=False)
                os.makedirs(self.dumpdir)
                if self.cfg.parallel:
                    os.makedirs(self.dumpdir_static)
                    os.makedirs(self.dumpdir_dynamic)
        else:
            if not os.path.isdir(self.dumpdir):
                os.makedirs(self.dumpdir)
            if self.is_master():
                remove_dir(self.dumpdir_dynamic, strict=False)
                remove_dir(self.hls_dumpdir_dynamic, strict=False)
                os.makedirs(self.dumpdir_dynamic)
                os.makedirs(self.dumpdir_dynamic + "/.graph_dumps/")
                os.makedirs(self.hls_dumpdir_dynamic)
            elif self.is_slave():
                remove_dir(self.dumpdir_static, strict=False)
                remove_dir(self.hls_dumpdir_static, strict=False)
                os.makedirs(self.dumpdir_static)
                os.makedirs(self.dumpdir_static + "/.graph_dumps/")
                os.makedirs(self.hls_dumpdir_static)

        remove_dir(self.logdir, strict=False)
        os.makedirs(self.logdir)

        self.init_logger()

    def init_logger(self):
        outfile = self.logdir + "/analyzer_out.log"
        self.logfile = open(outfile, "w")

    def __del__(self):
        if hasattr(self, "logfile") and not self.logfile.closed:
            self.logfile.close()

    def log(self, message, console=True):
        if console:
            print(message)
        self.logfile.write(f"{message}\n")

    def validate_dump_path(self):
        if self.cfg.parallel:
            assert os.path.exists(self.dumpdir_static), "Static dumps not found"
            assert os.path.exists(self.dumpdir_dynamic), "Dynamic dumps not found"

        else:
            assert os.path.exists(os.path.join(self.dumpdir, "./StaticSynRec.db")), "Static dumps not found"
            assert os.path.exists(os.path.join(self.dumpdir, "./DynamicSynRec.db")), "Dynamic dumps not found"

    @staticmethod
    def get_synrec_path():
        possible_paths = []
        if os.environ.get("SYNAPSE_ROOT"):
            possible_paths.append(os.path.join(os.environ.get("SYNAPSE_ROOT"), "scripts", "synrec.py"))
        possible_paths.append("/root/repos/synapse/scripts/synrec.py")
        possible_paths.append("/root/synapse/scripts/synrec.py")
        possible_paths.append("/software/synrec/synrec.py")

        for path in possible_paths:
            if os.path.isfile(path):
                return path
        raise RuntimeError(f'[ERROR] Could not find "synrec.py" from {possible_paths}')

    @staticmethod
    def get_json_tests_bin():
        possible_paths = []

        if os.environ.get("SYNAPSE_RELEASE_BUILD"):
            possible_paths.append(os.path.join(os.environ.get("SYNAPSE_RELEASE_BUILD"), "bin", "json_tests"))
        possible_paths.append(os.path.join("/software/users/dsa_bins/json_tests"))

        for path in possible_paths:
            if os.path.isfile(path):
                return path
        raise RuntimeError(f'[ERROR] Could not find "json_tests" from {possible_paths}')

    def get_commands(self):
        synrec_path = self.get_synrec_path()
        default_config = "PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1"
        if self.cfg.cache == 0:
            default_config = default_config + " PT_HPU_PGM_ENABLE_CACHE=0"
        do_split = " -s" if self.cfg.parallel else ""
        ranks = "--ranks " + str(self.cfg.rank)
        dump_dir_static = self.dumpdir_static
        dump_dir_dynamic = self.dumpdir_dynamic
        if self.is_master_slave_config():
            dump_dir_static = self.hls_dumpdir_static
            dump_dir_dynamic = self.hls_dumpdir_dynamic
        cmd_static = f"{default_config} PT_HPU_ENABLE_MIN_MAX_AS_CURRENT=1 {synrec_path}{do_split} -t -p {dump_dir_static} --ignore-errors --overwrite {ranks} -- {self.cfg.cmd}"
        cmd_dynamic = f"{default_config} {synrec_path}{do_split} -t -p {dump_dir_dynamic} --ignore-errors --overwrite {ranks} -- {self.cfg.cmd}"
        return cmd_static, cmd_dynamic

    def clear_cache(self):
        self.dict_cache = None

    def collect_available_dumps(self):
        def fill_dict(data_dict, path, mode):
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(".db"):
                        graph_name = file.split(".")[0]
                        prcoess_id = file.split(".")[1]
                        path_db = f"{path}/{graph_name}.{prcoess_id}.db"
                        path_json = f"{path}/{graph_name}.{prcoess_id}.json"
                        dict_key = ".graph_dumps/" + graph_name
                        data_dict[mode][dict_key] = {
                            "db": path_db,
                            "json": path_json,
                        }

        if self.use_cache and self.dict_cache is not None:
            return self.dict_cache
        else:
            data_dict = {"Static": {}, "Dynamic": {}}

            if self.cfg.parallel:
                graphdir_static = self.dumpdir_static + "/.graph_dumps/"
                fill_dict(data_dict, graphdir_static, "Static")
                graphdir_dynamic = self.dumpdir_dynamic + "/.graph_dumps/"
                fill_dict(data_dict, graphdir_dynamic, "Dynamic")

            else:
                static_db = self.dumpdir + "/StaticSynRec.db"
                static_json = self.dumpdir + "/StaticSynRec.json"
                dynamic_db = self.dumpdir + "/DynamicSynRec.db"
                dynamic_json = self.dumpdir + "/DynamicSynRec.json"
                data_dict["Static"] = {"db": static_db, "json": static_json}
                data_dict["Dynamic"] = {"db": dynamic_db, "json": dynamic_json}

            if self.use_cache:
                self.dict_cache = data_dict

            return data_dict

    def is_master_slave_config(self):
        if self.cfg.master or self.cfg.slave:
            return True
        return False

    def is_master(self):
        return self.cfg.master

    def is_slave(self):
        return self.cfg.slave

    # Not moving 2 files, because last 2 files can still be in writing
    # .json and .db
    def move_files(self, source_dir, destination_dir, move_all=False):
        # Get a list of all files in the source directory
        source_dir = source_dir + "/.graph_dumps/"
        destination_dir = destination_dir + "/.graph_dumps/"
        if not os.path.exists(source_dir):
            return

        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        valid_files = [f for f in files if f.endswith((".db", ".json"))]
        # Sort valid files by modification time
        valid_files.sort(key=lambda x: os.path.getmtime(os.path.join(source_dir, x)))

        # Move all valid files except the latest one based on the flag
        for file in valid_files if move_all else valid_files[:-2]:
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)

    def compare_databases(self, db_static, db_dynamic):
        assert not (os.path.getsize(db_static) == 0), f"db file {db_static} is empty"
        assert not (os.path.getsize(db_dynamic) == 0), f"db file {db_dynamic} is empty"

        conn1 = sqlite3.connect(db_static)
        conn2 = sqlite3.connect(db_dynamic)

        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

        # Get list of tables in both databases
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables1 = cursor1.fetchall()
        cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables2 = cursor2.fetchall()

        """
        This is thr format in which data is preset in DB file
        Data is from synapse/src/data_serialize/sql/sql_db_serializer.cpp
                                        "ROW_INDEX      int     not NULL,"
                                        "GROUP_ID       int     not NULL,"
                                        "LAUNCH_INDEX   int     not NULL,"
                                        "GRAPH_NAME     text    not NULL,"
                                        "RECIPE_ID      int     not NULL,"
                                        "NAME           text    not NULL,"
                                        "ID             int     not NULL,"
                                        "ITERATION      int     not NULL,"
                                        "TYPE           int     not NULL,"
                                        "DATA_TYPE      int     not NULL,"
                                        "VALIDATION     int     not NULL,"
                                        "CONST_TENSOR   int     not NULL,"
                                        "SHAPE          blob,"
                                        "PERMUTATION    blob,"
                                        "DATA_ID        int     not NULL,"
        """
        idx_graph_name = 3
        idx_tensor_name = 5
        idx_validation = 10
        idx_data = 14
        idx_iter = 7
        idx_launch = 2

        # Check whether the data in each table is the same in both databases
        cursor1.execute(f"SELECT * FROM TENSORS")
        tensors_static = cursor1.fetchall()
        cursor2.execute(f"SELECT * FROM TENSORS")
        tensors_dynamic = cursor2.fetchall()

        compare_len = len(tensors_dynamic)
        if len(tensors_static) != len(tensors_dynamic):
            self.log(
                "[WARNING] DB has different tensor numbers in static and dynamic only comparing the common ones",
                console=True,
            )
            compare_len = min(len(tensors_static), len(tensors_dynamic))

        # Validate if tensor names match for all common entries
        tensor_names_static = [item[idx_tensor_name] for item in tensors_static[:compare_len]]
        tensor_names_dynamic = [item[idx_tensor_name] for item in tensors_dynamic[:compare_len]]
        if not tensor_names_static == tensor_names_dynamic:
            self.log("[ERROR] DB has different Tensor name for tensor in static and dynamic not comparing.")
            if self.cfg.cache == True:
                self.log("[ERROR] Check with --cache 0.")
            exit(0)

        data_ids_table1 = [item[idx_data] for item in tensors_static[:compare_len]]
        data_ids_table2 = [item[idx_data] for item in tensors_dynamic[:compare_len]]
        if data_ids_table1 != data_ids_table2:
            for t_static, t_dynamic in zip(tensors_static[:compare_len], tensors_dynamic[:compare_len]):
                # Check if tensor is valid and data is different
                if (
                    (t_static[idx_validation] == 0)
                    and (t_dynamic[idx_validation] == 0)
                    and (t_static[idx_data] != t_dynamic[idx_data])
                ):
                    graph_name = t_dynamic[idx_graph_name]
                    if "graph_dumps" in graph_name:
                        if self.mismatch_map is None:
                            self.mismatch_map = {}
                            self.mismatch_static = []
                            self.mismatch_dynamic = []
                        if graph_name not in self.mismatch_map.keys():
                            self.mismatch_map[graph_name] = set()
                            self.log(f"[INFO] Mismatch found in graph {graph_name}")
                        self.mismatch_map[graph_name].add(t_dynamic[idx_tensor_name])
                        self.mismatch_static.append(
                            [t_static[idx_graph_name], t_static[idx_tensor_name], t_static[idx_iter]]
                        )
                        self.mismatch_dynamic.append(
                            [t_dynamic[idx_graph_name], t_dynamic[idx_tensor_name], t_dynamic[idx_iter]]
                        )

    def dump_stats(self):
        if self.mismatch_map is not None:
            outfile = self.logdir + "/mismatch.txt"
            self.log(f"[WARNING] Divergence in {len(self.mismatch_map)} graphs between static and dynamic runs")
            self.log(f"[INFO] Dumping divergence data in file\033[91m {outfile}\033[0m")
            with open(outfile, "w") as file:
                for key in self.mismatch_map:
                    file.write(f"{key} {self.mismatch_map[key]}\n")
        else:
            self.log("[INFO] The static and dynamic runs are equal")

    @staticmethod
    def read_values(path):
        with open(path, "r") as file:
            # skip first line as it contains tensor name
            return np.float64(file.read().strip().split("\n")[1:])

    @staticmethod
    def compare_values(values_static, values_dynamic):
        stats = {}
        stats.update(calc_difference(values_static, values_dynamic))
        stats.update(calc_similarity(values_static, values_dynamic))
        return stats

    @staticmethod
    def get_node(path, graph_name, tensor):
        with open(path, "r") as file:
            json_data = json.load(file)
        for graph in json_data["graphs"]:
            if graph["name"] == graph_name:
                for node in graph["nodes"]:
                    if tensor in node["output_tensors"]:
                        return {
                            "Graph": graph_name,
                            "Tensor": tensor,
                            "I/O": "output",
                            "node": node["name"],
                            "guid": node["guid"],
                        }
                    elif tensor in node["input_tensors"]:
                        return {
                            "Graph": graph_name,
                            "Tensor": tensor,
                            "I/O": "input",
                            "node": node["name"],
                            "guid": node["guid"],
                        }

        return {"Graph": graph_name, "Tensor": tensor, "I/O": None, "node": None, "guid": None}

    def dump_csv(self):
        if self.mismatch_map is None:
            return

        def get_path(data_dict, graph_name):
            return (
                data_dict["Static"][graph_name]["db"],
                data_dict["Dynamic"][graph_name]["db"],
                data_dict["Static"][graph_name]["json"],
            )

        path_csv = self.logdir + "/synrec_comparision.csv"
        if not self.cfg.no_stats:
            self.log(f"[INFO] Analyzing differences using dbparser and dumping in CSV file \033[91m{path_csv}\033[0m")
        else:
            self.log(f"[INFO] Dumping difference in CSV file \033[91m{path_csv}\033[0m")
        data_dict = self.collect_available_dumps()
        output_static = self.logdir + "/output_static.log"
        output_dynamic = self.logdir + "/output_dynamic.log"

        rows = []

        def process_outputs():
            values_static = self.read_values(output_static)
            values_dynamic = self.read_values(output_dynamic)
            stats = self.compare_values(values_static, values_dynamic)

            remove_file(output_static, verbose=False)
            remove_file(output_dynamic, verbose=False)

            return stats

        json_tests_bin = self.get_json_tests_bin()
        total_mismatches = sum([len(item) for item in self.mismatch_map.values()])
        progbar = tqdm.tqdm(total=total_mismatches)

        csv_outfile = open(path_csv, "w")
        is_first_row = True

        for static_list, dynamic_list in zip(self.mismatch_static, self.mismatch_dynamic):
            _graph_name_dynamic = dynamic_list[0]
            tensor_dynamic = dynamic_list[1]
            iteration_dynamic = dynamic_list[2]
            _graph_name_static = static_list[0]
            tensor_static = static_list[1]
            iteration_static = static_list[2]
            self.log(f'Analyzing graph:", {_graph_name_dynamic}, "-> Tensor:", {tensor_dynamic}', console=False)
            self.log(
                f'\tDynamic graph:", {_graph_name_dynamic}, "-> Tensor:", {tensor_dynamic}, " -> iteration: ", {iteration_dynamic}',
                console=False,
            )
            self.log(
                f'\tStatic  graph:", {_graph_name_static}, "-> Tensor:", {tensor_static}, " -> iteration: ", {iteration_static}',
                console=False,
            )
            if self.cfg.parallel:
                path_static, path_dynamic, path_json = get_path(data_dict, _graph_name_dynamic)
            else:
                path_static = data_dict["Static"]["db"]
                path_dynamic = data_dict["Dynamic"]["db"]
                path_json = data_dict["Static"]["json"]

            if not self.cfg.no_stats:
                # FIXME: Check if command ran successfully and found the graph and tensor in db file
                cmd_static = f"{json_tests_bin} db_parser -d {path_static} -g {_graph_name_static} -t {tensor_static} -i {iteration_static} -o {output_static}"
                cmd_dynamic = f"{json_tests_bin} db_parser -d {path_dynamic} -g {_graph_name_dynamic} -t {tensor_dynamic} -i {iteration_dynamic} -o {output_dynamic}"

                self.run(cmd_static, mode="static", verbose=False)
                self.run(cmd_dynamic, mode="dynamic", verbose=False)

            row = {}
            row.update(self.get_node(path_json, _graph_name_dynamic, tensor_dynamic))
            if not self.cfg.no_stats:
                row.update(process_outputs())

            if is_first_row:
                writer = csv.DictWriter(csv_outfile, row.keys())
                writer.writeheader()
                is_first_row = False
            if not self.cfg.no_stats and row["abs_max"] > self.cfg.thresh:
                writer.writerow(row)

            progbar.update(1)

        csv_outfile.close()

    def compare_dumps(self):
        data_dict = self.collect_available_dumps()
        if self.cfg.parallel:
            data_static = data_dict["Static"]
            data_dynamic = data_dict["Dynamic"]
            assert len(set(data_static) - set(data_dynamic)) == 0
            graph_names = list(sorted(data_static.keys(), key=lambda item: int(item.split("_")[-1])))

            for graph_name in graph_names:
                self.compare_databases(data_static[graph_name]["db"], data_dynamic[graph_name]["db"])
        else:
            db_static = data_dict["Static"]["db"]
            db_dynamic = data_dict["Dynamic"]["db"]
            self.compare_databases(db_static, db_dynamic)

    def compare_split(self, is_final=True):
        data_dict = self.collect_available_dumps()
        data_static = data_dict["Static"]
        data_dynamic = data_dict["Dynamic"]

        valid_files_count = min(len(data_static), len(data_dynamic))
        if not is_final:
            valid_files_count -= 1
        static_files = set(sorted(data_static.keys(), key=lambda item: int(item.split("_")[-1])))
        dynamic_files = set(sorted(data_dynamic.keys(), key=lambda item: int(item.split("_")[-1])))
        common_files = sorted(static_files.intersection(dynamic_files))
        graph_names = list(common_files)[:valid_files_count]
        self.log(f"[INFO] Comparing Graphs: {graph_names}", console=False)
        for graph_name in graph_names:
            self.compare_databases(data_static[graph_name]["db"], data_dynamic[graph_name]["db"])

        self.log(f"[INFO] Valid files count: {valid_files_count}", console=False)

        def delete_file(graph_name):
            self.log(f"[INFO] Deleting files of graph name: {graph_name} - Static and Dynamic matched", console=False)
            remove_file(data_static[graph_name]["db"], verbose=False)
            remove_file(data_static[graph_name]["json"], verbose=False)
            remove_file(data_dynamic[graph_name]["db"], verbose=False)
            remove_file(data_dynamic[graph_name]["json"], verbose=False)

        for graph_name in graph_names:
            if self.mismatch_map is None:
                delete_file(graph_name)
            else:
                is_mismatch_graph = any(graph_name in key for key in self.mismatch_map.keys())
                if not is_mismatch_graph:
                    delete_file(graph_name)

        if self.use_cache:
            self.clear_cache()

    def run(self, cmd, mode, verbose=True):
        outfile = f"{self.logdir}/{mode}_out.txt"

        if verbose:
            self.log(f"[INFO] Running in [{mode} mode] {cmd}")
        else:
            outfile = "/dev/null"

        cmd_full = f'script -e -q -c "{cmd}" {outfile} > /dev/null'
        status = os.system(cmd_full)
        assert status == 0, f"[ERROR] Dumping error logs to\033[91m {outfile}\033[0m"

    def train(self):
        cmd_static, cmd_dynamic = self.get_commands()
        p1 = mp.Process(target=self.run, args=(cmd_static, "static"))
        p1.start()
        p1.join()
        p1.close()
        p2 = mp.Process(target=self.run, args=(cmd_dynamic, "dynamic"))
        p2.start()
        p2.join()
        p2.close()
        self.log(f"[INFO] Finished training.\n       dumps: {self.dumpdir}\n       logs : {self.logdir}")

    def compare(self):
        self.compare_dumps()
        self.dump_stats()

    def run_train_commands_and_compare_parallel(self, cmd_static, cmd_dynamic, verbose=True):
        graphdir_static = self.dumpdir_static + "/.graph_dumps/"
        graphdir_dynamic = self.dumpdir_dynamic + "/.graph_dumps/"

        p1 = mp.Process(target=self.run, args=(cmd_static, "static", verbose))
        p2 = mp.Process(target=self.run, args=(cmd_dynamic, "dynamic", verbose))

        def _await(exit_gracefully=True):
            os.system("reset")  # FIXME: The "script" command messes up the terminal.
            p1.join()
            p2.join()
            if exit_gracefully:
                assert p1.exitcode == 0
                assert p2.exitcode == 0
            p1.close()
            p2.close()

        p1.start()
        p2.start()

        while p1.is_alive() or p2.is_alive():
            if os.path.exists(graphdir_static) and os.path.exists(graphdir_dynamic):
                time.sleep(5)
                self.compare_split(is_final=False)
                if self.mismatch_map is not None and self.cfg.eam <= len(self.mismatch_map.keys()):
                    p1.kill()
                    p2.kill()
                    _await(exit_gracefully=False)
                    return

        _await(exit_gracefully=True)
        self.compare_split(is_final=True)

    def train_and_compare_parallel(self):
        cmd_static, cmd_dynamic = self.get_commands()
        if not self.is_master_slave_config():
            self.run_train_commands_and_compare_parallel(cmd_static, cmd_dynamic)
            self.dump_stats()
            self.log(f"[INFO] Finished training.\n       dumps: {self.dumpdir}\n       logs : {self.logdir}")
        elif self.is_master():
            p1 = mp.Process(target=self.run, args=(cmd_dynamic, "dynamic", True))
            p1.start()
            while p1.is_alive():
                time.sleep(5)
                self.move_files(self.hls_dumpdir_dynamic, self.dumpdir_dynamic, False)
                self.compare_split(is_final=False)
                if self.mismatch_map is not None and self.cfg.eam <= len(self.mismatch_map.keys()):
                    p1.kill()
            p1.join()
            p1.close()
            self.move_files(self.hls_dumpdir_dynamic, self.dumpdir_dynamic, True)
            finished_file_path = os.path.join(self.dumpdir_static, "Finished")
            while not os.path.isfile(finished_file_path):
                time.sleep(10)
            self.compare_split(is_final=True)
            self.dump_stats()
            self.log(f"[INFO] Finished training dynamic.\n       dumps: {self.dumpdir}\n       logs : {self.logdir}")
        elif self.is_slave():
            p1 = mp.Process(target=self.run, args=(cmd_static, "static", True))
            p1.start()
            while p1.is_alive():
                time.sleep(5)
                self.move_files(self.hls_dumpdir_static, self.dumpdir_static, False)
            p1.join()
            p1.close()
            self.move_files(self.hls_dumpdir_static, self.dumpdir_static, True)
            # Create a file named "finished" in the Static directory acts as sync between static and dynamic
            finished_file_path = os.path.join(self.dumpdir_static, "Finished")
            with open(finished_file_path, "w") as finished_file:
                finished_file.write("Static finished execution.")
            self.log("[INFO] Finished training Static.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd",
        type=str,
        required=False,
        help="If Specified run the command on the device in static and dynamic and do a comparison, command to be specified in quotes",
    )
    parser.add_argument(
        "--out", type=str, default="/tmp/dumps", help="The output directory to dump or read the dumps from"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        required=False,
        help="Add this flag to run in parallel mode, Dynamic in 1 process, and Static in another process. The dumps are compared on the fly and deleted if matching.",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=1,
        help="Configure the enablement/disablement of recipe cache. In parallel/8x disabled by default",
    )
    parser.add_argument(
        "--thresh", type=float, default=0.001, help="The threshold above which dumps the stats in CSV file"
    )
    parser.add_argument(
        "--slave", action="store_true", required=False, help="If specified, runs the Static command in slave mode"
    )
    parser.add_argument(
        "--master",
        action="store_true",
        required=False,
        help="If specified, runs the Dynamic command in master mode, dumping and comparision",
    )
    parser.add_argument(
        "--csv", type=int, default=1, help="Enabled by default, dumps the differences/Stats in the CSV file "
    )
    parser.add_argument(
        "--eam",
        type=int,
        default=10000,
        help="After finding the first difference continue to dump eam number of graphs more",
    )
    parser.add_argument("--rank", type=int, default=0, help="Rank to record in multi card case, default 0")
    parser.add_argument(
        "--no_stats",
        action="store_true",
        required=False,
        help="If specified, dont dump stats, no db-parser_required, All tensors with hash diff dumped",
    )
    parser.add_argument(
        "--print_cmd", action="store_true", required=False, help="If specified, only print the synrec command"
    )

    args = parser.parse_args()
    valid_ints = {0, 1}
    assert args.csv in valid_ints

    return args


def main(args):
    divergence_analyzer = DivergenceAnalyzer(args)
    if args.print_cmd:
        cmd_static, cmd_dynamic = divergence_analyzer.get_commands()
        print("Static CMD - \033[92m", cmd_static, "\033[0m")
        print("Dynamic CMD - \033[91m", cmd_dynamic, "\033[0m")
        return

    if divergence_analyzer.is_master_slave_config():
        args.parallel = 1

    if args.parallel == 1:
        args.cache = 0

    if args.cmd is not None:
        if args.parallel:
            import habana_frameworks.torch.hpu as hpu

            if hpu.device_count() < 2:
                print(f"[ERROR]: Found only {hpu.device_count()} HPU device(s). Cannot running in parallel mode.")
                return
            divergence_analyzer.train_and_compare_parallel()
        else:
            divergence_analyzer.train()
            divergence_analyzer.compare()
    else:
        divergence_analyzer.compare()
    if args.csv:
        divergence_analyzer.dump_csv()


if __name__ == "__main__":
    args = get_args()
    main(args)
