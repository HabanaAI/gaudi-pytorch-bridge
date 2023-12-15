import argparse
import csv
import json
import multiprocessing as mp
import numpy as np
import os
import shutil
import sqlite3
import time
import tqdm

def remove_file(path, verbose=True, strict=True):
    if os.path.isfile(path):
        if verbose:
            print(f'[INFO] Deleting file {path}')
        os.remove(path)
    else:
        if strict:
            raise ValueError(f'{path} is not a file')

def remove_dir(path, strict=True):
    if os.path.isdir(path):
        print(f'[INFO] Deleting directory {path}')
        shutil.rmtree(path)
    else:
        if strict:
            raise ValueError(f'{path} is not a directory')

def calc_difference(a, b):
    diff = np.subtract(a, b)
    abs_diff = np.abs(diff)
    abs_max = np.amax(abs_diff)
    abs_min = np.amin(abs_diff)

    mse = np.mean(np.square(diff))
    rmse = np.sqrt(mse)
    return {'abs_max': abs_max.item(), 'abs_min': abs_min.item(), 'mse': mse.item(), 'rmse': rmse.item()}

def calc_similarity(a, b, threshold=0.0,):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    norm_relative = np.divide(norm_a, norm_b)
    angle = np.arccos(min(np.dot(a, b) / norm_a / norm_b, 1.0)) / np.pi * 180
    angle = np.around(angle, 2)
    cosine_similarity = np.less_equal(angle, threshold)
    all_close = np.allclose(a, b)
    return {'norm_a': norm_a.item(), 'norm_b': norm_b.item(), 'norm_relative': norm_relative.item(), 'angle': angle.item(), 'cosine_similarity': cosine_similarity, 'all_close': all_close}

class DivergenceAnalyzer:
    def __init__(self, cfg, use_cache=True):
        self.cfg = cfg
        self.dumpdir = os.path.join(args.outdir, 'dumps')
        self.logdir = os.path.join(self.dumpdir, 'logs')
        self.dumpdir_static = os.path.join(self.dumpdir, 'StaticSynRec')
        self.dumpdir_dynamic = os.path.join(self.dumpdir, 'DynamicSynRec')
        self.dict_cache = None
        self.mismatch_map = None
        self.use_cache = use_cache
        self.verify_cfg()

        if self.cfg.do_train: # Start anew for training.
            remove_dir(self.dumpdir, strict=False)
        remove_dir(self.logdir, strict=False)
        os.makedirs(self.logdir)

        self.init_logger()

    def init_logger(self):
        outfile = self.logdir + '/analyzer_out.log'
        self.logfile = open(outfile, 'w')

    def __del__(self):
        if hasattr(self, 'logfile') and not self.logfile.closed:
            self.logfile.close()

    def log(self, message, console=True):
        if console:
            print(message)
        self.logfile.write(f'{message}\n')

    def verify_cfg(self):
        assert self.cfg.outdir, '[ERROR] Specify directory to dump the outputfiles. Got empty string.'
        if self.cfg.do_train:
            assert self.cfg.train_cmd, '[ERROR] Specify workload command for training in train_cmd. Got empty string.'
            if self.cfg.enable_parallel:
                assert self.cfg.do_split, '[ERROR] Enable do_split when using parallel mode.'
        else:
            if not self.cfg.do_compare:
                if not self.cfg.to_csv:
                    print('[INFO] Nothing to compute.')
                    exit(0)
            self.validate_dump_path()
        if self.cfg.to_csv:
            assert self.cfg.do_compare, '[ERROR] Enable do_compare when using to_csv.'

    def validate_dump_path(self):
        if self.cfg.do_split:
            assert os.path.exists(self.dumpdir_static), 'Static dumps not found'
            assert os.path.exists(self.dumpdir_dynamic), 'Dynamic dumps not found'

        else:
            assert os.path.exists(os.path.join(self.dumpdir, './StaticSynRec.db')), 'Static dumps not found'
            assert os.path.exists(os.path.join(self.dumpdir, './DynamicSynRec.db')), 'Dynamic dumps not found'

    @staticmethod
    def get_synrec_path():
        possible_paths = [
            "/root/repos/synapse/scripts/synrec.py",
            "/root/synapse/scripts/synrec.py",
        ]

        if os.environ.get('SYNAPSE_ROOT'):
            possible_paths.append(
                os.path.join(
                    os.environ.get('SYNAPSE_ROOT'),
                    "scripts",
                    "synrec.py"
                )
            )

        for path in possible_paths:
            if os.path.isfile(path):
                return path
        raise RuntimeError(f'[ERROR] Could not find "synrec.py" from {possible_paths}')

    @staticmethod
    def get_json_tests_bin():
        possible_paths = []

        if os.environ.get('SYNAPSE_RELEASE_BUILD'):
            possible_paths.append(
                os.path.join(
                    os.environ.get('SYNAPSE_RELEASE_BUILD'),
                    "bin",
                    "json_tests"
                )
            )

        for path in possible_paths:
            if os.path.isfile(path):
                return path
        raise RuntimeError(f'[ERROR] Could not find "json_tests" from {possible_paths}')

    def get_commands(self):
        synrec_path = self.get_synrec_path()
        default_config = "PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1"
        do_split = ' -s' if self.cfg.do_split else ''
        hpu_mode = 0 if self.cfg.run_eager_mode else 1

        cmd_static = f'{default_config} PT_HPU_LAZY_MODE={hpu_mode} PT_HPU_ENABLE_MIN_MAX_AS_CURRENT=1 {synrec_path}{do_split} -t -p {self.dumpdir_static} --ignore-errors --overwrite -- {self.cfg.train_cmd}'
        cmd_dynamic = f'{default_config} PT_HPU_LAZY_MODE={hpu_mode} {synrec_path}{do_split} -t -p {self.dumpdir_dynamic} --ignore-errors --overwrite -- {self.cfg.train_cmd}'

        return cmd_static, cmd_dynamic

    def clear_cache(self):
        self.dict_cache = None

    def collect_available_dumps(self):
        def fill_dict(data_dict, path, mode):
            for file in os.listdir(path):
                if file.endswith(".db"):
                    graph_name = file.split('.')[0]
                    prcoess_id = file.split('.')[1]
                    path_db = f'{path}/{graph_name}.{prcoess_id}.db'
                    path_json = f'{path}/{graph_name}.{prcoess_id}.json'
                    data_dict[mode][graph_name] = {
                        'db': path_db,
                        'json': path_json,
                    }

        if self.use_cache and self.dict_cache is not None:
            return self.dict_cache
        else:
            data_dict = {'Static': {}, 'Dynamic': {}}

            if self.cfg.do_split:
                graphdir_static = self.dumpdir_static + '/.graph_dumps/'
                fill_dict(data_dict, graphdir_static, 'Static')

                graphdir_dynamic = self.dumpdir_dynamic + '/.graph_dumps/'
                fill_dict(data_dict, graphdir_dynamic, 'Dynamic')

            else:
                static_db = self.dumpdir + '/StaticSynRec.db'
                static_json = self.dumpdir + '/StaticSynRec.json'
                dynamic_db = self.dumpdir + '/DynamicSynRec.db'
                dynamic_json = self.dumpdir + '/DynamicSynRec.json'
                data_dict['Static'] = {'db': static_db, 'json': static_json}
                data_dict['Dynamic'] = {'db': dynamic_db, 'json': dynamic_json}

            if self.use_cache:
                self.dict_cache = data_dict

            return data_dict

    def compare_databases(self, db_file1, db_file2):
        assert not (os.path.getsize(db_file1) == 0), f'db file {db_file1} is empty'
        assert not (os.path.getsize(db_file2) == 0), f'db file {db_file2} is empty'

        conn1 = sqlite3.connect(db_file1)
        conn2 = sqlite3.connect(db_file2)

        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

        # Get list of tables in both databases
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables1 = cursor1.fetchall()
        cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables2 = cursor2.fetchall()

        # Check whether the number of tables is the same in both databases
        if len(tables1) != len(tables2):
            self.log("[WARNING] Databases have different number of tables")
            return False

        # Check whether the tables have the same names in both databases
        if sorted(tables1) != sorted(tables2):
            self.log("[WARNING] Databases have different table names")
            return False

        '''
        This is thr format in which data is preset in DB file
        Data is from synapse/src/data_serialize/sql_db_serializer.cpp
            0 "ROW_INDEX      int     not NULL,"
            1 "GRAPH_NAME     text    not NULL,"
            2 "RECIPE_ID      int     not NULL,"
            3 "NAME           text    not NULL,"
            4 "ID             int     not NULL,"
            5 "ITERATION      int     not NULL,"
            6 "TYPE           int     not NULL,"
            7 "DATA_TYPE      int     not NULL,"
            8 "COMPRESSION    int     not NULL,"
            9 "VALIDATION     int     not NULL,"
            10 "CONST_TENSOR   int     not NULL,"
            11 "SHAPE          blob,"
            12 "PERMUTATION    blob,"
            13 "DATA_ID        int     not NULL,"
            14 "CONSTRAINT tensor_pk PRIMARY KEY (RECIPE_ID,ROW_INDEX,ID,ITERATION),"
            15 "FOREIGN KEY(DATA_ID) references DATA(ID));",
        '''
        idx_name = 3
        idx_validation = 9
        idx_data = 13

        # Check whether the data in each table is the same in both databases
        for table_name in tables1[2:]:
            cursor1.execute(f"SELECT * FROM \"{table_name[0]}\"")
            rows_table1 = cursor1.fetchall()
            cursor2.execute(f"SELECT * FROM \"{table_name[0]}\"")
            rows_table2 = cursor2.fetchall()
            if len(rows_table1) != len(rows_table2):
                self.log("[WARNING]", table_name[0] + " has different tensor numbers in static and dynamic not comparing")
            else:
                data_ids_table1 = [item[idx_data] for item in rows_table1]
                data_ids_table2 = [item[idx_data] for item in rows_table2]

                if data_ids_table1 != data_ids_table2:
                    if self.mismatch_map is None:
                        self.mismatch_map = {}
                    graph_name = rows_table1[0][1]
                    self.mismatch_map[graph_name] = []
                    for r1, r2 in zip(rows_table1, rows_table2):
                        # Check if tensor is valid and data is different
                        if((r1[idx_validation] == 0) and (r2[idx_validation] == 0 ) and (r1[idx_data] != r2[idx_data])):
                            self.mismatch_map[graph_name].append(r1[idx_name])

    def compare_dumps(self):
        data_dict = self.collect_available_dumps()

        if self.cfg.do_split:
            data_static = data_dict['Static']
            data_dynamic = data_dict['Dynamic']

            assert len(set(data_static) - set(data_dynamic)) == 0
            graph_names = list(sorted(data_static.keys(), key=lambda item: int(item.split('_')[-1])))

            for graph_name in graph_names:
                self.compare_databases(data_static[graph_name]['db'], data_dynamic[graph_name]['db'])
        else:
            db_static = data_dict['Static']['db']
            db_dynamic = data_dict['Dynamic']['db']
            self.compare_databases(db_static, db_dynamic)

    def dump_stats(self):
        if self.mismatch_map is not None:
            outfile = self.dumpdir + '/mismatch.txt'
            self.log(f'[WARNING] Divergence in {len(self.mismatch_map)} graphs between static and dynamic runs')
            self.log(f'[INFO] Dumping divergence data in file\033[91m {outfile}\033[0m')
            with open(outfile, 'w') as file:
                for key in self.mismatch_map:
                    file.write(f'{key} {self.mismatch_map[key]}\n')
        else:
            self.log('[INFO] The static and dynamic runs are equal')

    @staticmethod
    def read_values(path):
        with open(path, 'r') as file:
            # skip first line as it contains tensor name
            return np.float64(file.read().strip().split('\n')[1:])

    @staticmethod
    def compare_values(values_static, values_dynamic):
        stats = {}
        stats.update(calc_difference(values_static, values_dynamic))
        stats.update(calc_similarity(values_static, values_dynamic))
        return stats

    @staticmethod
    def get_node(path, graph_name, tensor):
        with open(path, 'r') as file:
            json_data = json.load(file)
        for graph in json_data['graphs']:
            if graph['name'] == graph_name:
                for node in graph['nodes']:
                    if tensor in node['output_tensors']:
                        return {'Graph': graph_name, 'Tensor': tensor, 'I/O': 'output', 'node': node['name'], 'guid': node['guid']}
                    elif tensor in node['input_tensors']:
                        return {'Graph': graph_name, 'Tensor': tensor, 'I/O': 'input', 'node': node['name'], 'guid': node['guid']}

        return {'Graph': graph_name, 'Tensor': tensor, 'I/O': None, 'node': None, 'guid': None}

    def dump_csv(self):
        if self.mismatch_map is None:
            return

        def get_path(data_dict, graph_name):
            return data_dict['Static'][graph_name]['db'], data_dict['Dynamic'][graph_name]['db'], data_dict['Static'][graph_name]['json']

        path_csv = self.dumpdir + '/synrec_comparision.csv'

        self.log(f'[INFO] Analyzing differences using dbparser and dumping in CSV file \033[91m{path_csv}\033[0m')

        data_dict = self.collect_available_dumps()
        output_static = 'output_static.log'
        output_dynamic = 'output_dynamic.log'
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

        csv_outfile = open(path_csv, 'w')
        is_first_row = True

        for graph_name, tensors in self.mismatch_map.items():
            for tensor in tensors:
                _graph_name = graph_name.split('/')[-1]
                self.log(f'Analyzing graph:", {_graph_name}, "-> Tensor:", {tensor}', console=False)

                if self.cfg.do_split:
                    path_static, path_dynamic, path_json = get_path(data_dict, _graph_name)
                else:
                    path_static = data_dict['Static']['db']
                    path_dynamic = data_dict['Dynamic']['db']
                    path_json = data_dict['Static']['json']

                cmd_static = f"{json_tests_bin} db_parser -d {path_static} -g '{graph_name}' -t '{tensor}' -o {output_static}"
                cmd_dynamic = f"{json_tests_bin} db_parser -d {path_dynamic} -g '{graph_name}' -t '{tensor}' -o {output_dynamic}"

                self.run(cmd_static, mode='static', verbose=False)
                self.run(cmd_dynamic, mode='dynamic', verbose=False)

                row = {}
                row.update(self.get_node(path_json, graph_name, tensor))
                row.update(process_outputs())
                rows.append(row)

                if is_first_row:
                    writer = csv.DictWriter(csv_outfile, row.keys())
                    writer.writeheader()
                    is_first_row = False

                writer.writerow(row)

                progbar.update(1)

        csv_outfile.close()

    def run(self, cmd, mode, verbose=True):
        outfile = f'{self.logdir}/{mode}_out.txt'

        if verbose:
            self.log(f'[INFO] Running in [{mode} mode] {cmd}')
        else:
            outfile = '/dev/null'

        cmd_full = f'script -e -q -c "{cmd}" {outfile} > /dev/null'
        status = os.system(cmd_full)
        assert status == 0, f'[ERROR] Dumping error logs to\033[91m {outfile}\033[0m'

    def run_train_commands(self, cmd_static, cmd_dynamic, verbose=True):
        if self.cfg.enable_parallel:
            p1 = mp.Process(target=self.run, args=(cmd_static, 'static', verbose))
            p2 = mp.Process(target=self.run, args=(cmd_dynamic, 'dynamic', verbose))

            p1.start()
            p2.start()

            p1.join()
            p2.join()
            assert p1.exitcode == 0
            assert p2.exitcode == 0
            os.system('reset') # FIXME: The "script" command messes up the terminal.
            p1.close()
            p2.close()
        else:
            self.run(cmd_static, 'static', verbose)
            self.run(cmd_dynamic, 'dynamic', verbose)

    def run_train_commands_and_compare(self, cmd_static, cmd_dynamic, verbose=True):
        graphdir_static = self.dumpdir_static + '/.graph_dumps/'
        graphdir_dynamic = self.dumpdir_dynamic + '/.graph_dumps/'

        def _compare(is_final=True):
            data_dict = self.collect_available_dumps()
            data_static = data_dict['Static']
            data_dynamic = data_dict['Dynamic']

            valid_files_count = min(len(data_static), len(data_dynamic))
            if not is_final:
                valid_files_count -= 1

            graph_names = list(sorted(data_static.keys(), key=lambda item: int(item.split('_')[-1])))[:valid_files_count]
            for graph_name in graph_names:
                self.compare_databases(data_static[graph_name]['db'], data_dynamic[graph_name]['db'])

            self.log(f'[INFO] Valid files count: {valid_files_count}', console=False)

            if self.mismatch_map is None:
                for graph_name in graph_names:
                    self.log(f'[INFO] Deleting files of graph name: {graph_name}', console=False)
                    remove_file(data_static[graph_name]['db'], verbose=False)
                    remove_file(data_static[graph_name]['json'], verbose=False)
                    remove_file(data_dynamic[graph_name]['db'], verbose=False)
                    remove_file(data_dynamic[graph_name]['json'], verbose=False)

            if self.use_cache:
                self.clear_cache()

        p1 = mp.Process(target=self.run, args=(cmd_static, 'static', verbose))
        p2 = mp.Process(target=self.run, args=(cmd_dynamic, 'dynamic', verbose))

        def _await(exit_gracefully=True):
            os.system('reset') # FIXME: The "script" command messes up the terminal.
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
                _compare(is_final=False)
                if self.mismatch_map is not None and self.cfg.exit_on_first_mismatch:
                    p1.kill()
                    p2.kill()
                    _await(exit_gracefully=False)
                    return

        _await(exit_gracefully=True)
        _compare(is_final=True)

    def train(self):
        cmd_static, cmd_dynamic = self.get_commands()
        self.run_train_commands(cmd_static, cmd_dynamic)
        self.log(f'[INFO] Finished training.\n       dumps: {self.dumpdir}\n       logs : {self.logdir}')

    def train_and_compare(self):
        cmd_static, cmd_dynamic = self.get_commands()
        self.run_train_commands_and_compare(cmd_static, cmd_dynamic)
        self.dump_stats()
        self.log(f'[INFO] Finished training.\n       dumps: {self.dumpdir}\n       logs : {self.logdir}')

    def compare(self):
        self.compare_dumps()
        self.dump_stats()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cmd", type=str, help='The workload command to dump static and dynamic runs\' data')
    parser.add_argument("--do_train", type=int, default=1, help="If set to 1, enables training mode")
    parser.add_argument("--do_compare", type=int, default=1, help="If set to 1, compare synapse graphs/tensors")
    parser.add_argument("--do_split", type=int, default=1, help="If set to 1, split the synapse graphs/tensors to multiple files")
    parser.add_argument("--enable_parallel", type=int, default=1, help="If set to 0, forces single card training even when multiple devices are available")
    parser.add_argument("--to_csv", type=int, default=0, help="If set to 1, dumps the output to a CSV file in addition to the TXT file")
    parser.add_argument("--outdir", type=str, required=True, help="Specifies the directory to dump the output files")
    parser.add_argument("--exit_on_first_mismatch", type=int, default=0, help="If set to 1, program exits after encountering the very first graphs mismatch")
    parser.add_argument("--run_eager_mode", action='store_true', help='If specified, runs the workload in eager mode')

    args = parser.parse_args()
    valid_ints = {0, 1}
    assert args.do_train in valid_ints
    assert args.do_compare in valid_ints
    assert args.do_split in valid_ints
    assert args.enable_parallel in valid_ints
    assert args.to_csv in valid_ints
    assert args.exit_on_first_mismatch in valid_ints

    return args

def main(args):
    # Force non-parallel mode when device count < 2
    if args.enable_parallel:
        import habana_frameworks.torch.hpu as hpu # Avoid slow import if parallel mode is disabled.
        if hpu.device_count() < 2:
            args.enable_parallel = 0
            print(f'[WARNING]: Found only {hpu.device_count()} HPU device(s). Disabling parallel mode.')

    divergence_analyzer = DivergenceAnalyzer(args)

    if args.do_train and args.enable_parallel:
        if args.do_compare:
            divergence_analyzer.train_and_compare()
        else:
            divergence_analyzer.train()
    else:
        if args.do_train:
            divergence_analyzer.train()
        if args.do_compare:
            divergence_analyzer.compare()

    if args.to_csv:
        divergence_analyzer.dump_csv()

if __name__ == '__main__':
    args = get_args()
    main(args)
