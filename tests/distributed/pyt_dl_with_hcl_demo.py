#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import signal
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mpt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# global settings

# debuggerID is used to run one of the processes in debug mode
# this must be changed in the script to enable
debuggerID = -1

# 8 is the maximum processes per host
# this will be overridden by json configuration, if lower
num_processes = int(os.getenv("GAUDI_PROCESSES", 8))
print("----num_processes :: ", num_processes)
module_ids = []
test_params = {}
min_size_power = 0
max_size_power = 0
min_size_base = 0
max_size_base = 0
demo_exe = "./hcl_test"
perf_mode = False
print_log = lambda *x: None

test_list = (
    "all_reduce",
    "all_reduce_multi_streams",
    "all_reduce_comms_streams",
    "all2all",
    "all2all_v",
    "all2all_v_custom",
    "all_gather",
    "reduce_scatter",
    "read_write",
    "all",
    "loopback",
    "sanity",
    "compute_all_reduce",
)
test_usage = """ - 'all' will run all tests except 'sanity', 'loopback' and 'all2all_v_custom'
 - 'loopback' test requires HW dongle to set port loopback or habanalabs driver configuration.
    see README.md for details."""

try:
    os.environ["PT_HPU_LAZY_MODE"] = "1"
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"


def init_data_loader():
    wsize = 8  # mpi_comm.Get_size()
    rank = 0  # mpi_comm.Get_rank()
    # print(" world_size :: ", wsize)
    # print(" local rank :: ", rank)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_start_method("spawn", force=True)
    train_dir = pathlib.Path("/root/data/pytorch/imagenet/ILSVRC2012/")
    dataset = datasets.ImageFolder(train_dir, transform)
    bs = 256
    total_images = 1280000
    num_steps = total_images / bs / wsize
    workers = 8
    print("number of DL workers :: ", workers)
    print("number of steps :: ", num_steps)
    print("batch size :: ", bs)

    dl_type = "2"
    if len(sys.argv) >= 2:
        dl_type = sys.argv[1]

    if dl_type == "0":
        print("synthetic dataset selected")
        dataloader = ImageRandomDataLoader(batch_size=bs, num_steps=num_steps, train=True)
    elif dl_type == "1":
        try:
            import habana_torch_dataloader
        except ImportError:
            assert False, "Could Not import habana_torch_dataloader"
        print("Multi-Threading DL with imagenet dataset selected")
        dataloader = habana_torch_dataloader.DataLoader(dataset, batch_size=bs, num_workers=workers, shuffle=True)
    else:
        print("Multi-process DL with imagenet dataset selected")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=workers, shuffle=True)

    return dataloader, num_steps, bs, rank


def test_pytorch_data_loader_for_resnet(dataloader, num_steps, bs, rank):
    t_sum = 0
    t_ips = 0
    start_time = time.time()

    for i, (image, data) in enumerate(dataloader):
        dl_time = time.time()
        t_diff = dl_time - start_time
        t_sum += t_diff
        start_time = dl_time
        ips = bs / t_diff
        t_ips += ips

        # image, target = image.to('hpu', non_blocking=False), data.to('hpu', non_blocking=False)

        if rank == 0:
            print("iteration : ", i, " ips : ", ips)

        if i >= 100:  # 675: #num_steps:
            break

    print("Avg ips per card = ", t_ips / num_steps)
    print("Total time take = ", t_sum)


def start_dl_workers():
    workers = []
    for i in range(num_processes):
        w = mpt.Process(target=dl_process)
        w.daemon = False
        w.id = i
        w.start()
        workers.append(w)
    return workers


def stop_dl_workers(workers):
    for w in workers:
        w.join(timeout=100)
        if w.is_alive():
            w.terminate()


def dl_process():

    dataloader, num_steps, bs, rank = init_data_loader()
    test_pytorch_data_loader_for_resnet(dataloader, num_steps, bs, rank)


def show_test_list():
    print("Test list:")
    for test in test_list:
        print(f"    {test}")
    print(test_usage)


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_demo_cmd(
    id=0,
    size=1 << 5,
    inner_loop=20000,
    loop=10,
    json_file="hls1.json",
    max_comm_id=0,
    a2av_custom_file=None,
    test="all_reduce",
    data_type="bfloat16",
    size_range=None,
    size_range_base=None,
    same_address=False,
    no_weak_order=False,
    skip_correctness=False,
    skip_performance=False,
    prof=None,
    enable_console="false",
    isTee=False,
    isLog=False,
    gdb=-1,
    num_cores_per_socket=18,
    num_sockets=2,
    num_hyper_threads=2,
    proc_affinity=False,
):

    # mandatory args
    # keep ID first for readability
    demo_cmd = [
        f"ID={id}",
        f"SIZE={size}",
        f"INNER_LOOP={inner_loop}",
        f"LOOP={loop}",
        f"JSON={json_file}",
        f"STOP_ON_TIMEOUT=0",
        f"ENABLE_CONSOLE={enable_console}",
    ]

    if test == "all2all_v_custom":
        if a2av_custom_file is None:
            print("Error: No all2allv Custom Config provided. Aborting")
            sys.exit(1)
        demo_cmd.append(f"ALL2ALLV_CFG={a2av_custom_file}")

    if gdb >= 0 and id == gdb:
        demo_cmd.append("gdb --args")

    demo_cmd.extend([f"{demo_exe}", f"--data-type {data_type}", f"--test-name {test}"])

    if prof and ("all" in prof or str(id) in prof):
        # add to beginning of cmd, keep ID field first for readability
        demo_cmd.insert(1, "HABANA_PROFILE=1")
    if max_comm_id:
        demo_cmd.append(f"--max_comm_id {max_comm_id}")
    if size_range:
        demo_cmd.append(f"--size-range {size_range[0]} {size_range[1]}")
    if size_range_base:
        demo_cmd.append(f"--size-range-base {size_range_base[0]} {size_range_base[1]}")
    if same_address:
        demo_cmd.append(f"--same-address")
    if no_weak_order:
        demo_cmd.append(f"--no-weak-order")
    if skip_correctness:
        demo_cmd.append(f"--no-correctness")
    if skip_performance:
        demo_cmd.append(f"--no-performance")
    if num_cores_per_socket:
        demo_cmd.append(f"--num-cores-per-socket {num_cores_per_socket}")
    if num_sockets:
        demo_cmd.append(f"--num-sockets {num_sockets}")
    if num_hyper_threads:
        demo_cmd.append(f"--num-hyper-threads {num_hyper_threads}")
    if proc_affinity:
        demo_cmd.insert(1, "HCL_CPU_AFFINITY=1")
        demo_cmd.append(f"--proc-affinity")
    if isTee:
        demo_cmd.append(f"| tee {id}.txt")
    elif isLog:
        demo_cmd.append(f"> log{id}.txt")

    return " ".join(demo_cmd)


def run_process(p):
    print_log(f"Running: {p}")
    w = start_dl_workers()
    ecode = os.system(p)
    print("Attempting to load library pytorch plugin", flush=True)
    from habana_frameworks.torch.utils.library_loader import load_habana_module

    load_habana_module()

    hpu = torch.device("hpu")
    cpu = torch.device("cpu")
    cpu_tensor = torch.randn(256, 3, 224, 224)
    hpu_tensor = cpu_tensor.to(hpu)
    cpu_tensor.fill_(2020)
    hpu_tensor.fill_(2020)
    h2 = hpu_tensor.to(cpu)
    print(
        "Tensor values are same :: ",
        np.allclose(h2.detach().numpy(), cpu_tensor.detach().numpy(), atol=0.001, rtol=1.0e-3, equal_nan=True),
    )
    stop_dl_workers(w)
    # image = torch.randn(256, 3, 224, 224)
    # image.to('hpu', non_blocking=False)

    return ecode


def read_settings():
    global num_processes

    try:
        json_file_name = test_params["json_file"]
        datastore = read_json(json_file_name)
    except Exception as e:
        print("Error reading json file: {}. Reason: {}. Aborting".format(json_file_name, e))
        sys.exit(1)

    print_log("Printing json configuration file:")
    print_log(json.dumps(datastore, indent=2))
    if "HCL_RANKS" in datastore.keys():
        commSize = len(datastore["HCL_RANKS"])
    elif "HCL_COUNT" in datastore.keys():
        commSize = int(datastore["HCL_COUNT"])
    else:
        print("Failed to retrieve number of ranks from json file. Aborting")
        sys.exit(1)

    if test_params["test"] == "loopback" and commSize != 1:
        print("Configuration Error: loopback test is enabled only on one rank.")
        print("make sure test is configured properly (check JSON file).")
        sys.exit(1)

    if "HCL_TYPE" not in datastore.keys():
        print("Configuration Error: HCL_TYPE is not set. Aborting")
        sys.exit(1)

    if datastore["HCL_TYPE"] == "HLS1-H":
        num_processes = min(num_processes, 4)

    num_processes = min(num_processes, commSize)


def handle_make(isClean=False):
    make_cmd = "make"
    if isClean:
        make_cmd += " clean"
    elif "SYNAPSE_RELEASE_BUILD" in os.environ:
        print_log("Detected dev envirnoment!")
        make_cmd += " dev"
    run_process(make_cmd)


def clear_logs():
    rm_cmd = "rm -rf ~/.habana_logs*"
    run_process(rm_cmd)


def clean_artifacts():
    handle_make(isClean=True)
    clear_logs()
    all_files = os.listdir(".")
    files_to_delete = [f for f in all_files if f.endswith(".recipe.used") or f.endswith(".csv")]
    for f in files_to_delete:
        try:
            os.remove(f)
            print_log(f"Cleaning: {f}")
        except:
            print(f"Failed to remove file: {f}")


def configure_profiler():
    # enable user instrumentation and NICS in profiler configuration
    config_cmd = "hl-prof-config -gaudi -i -nic=on -add-pci=on"
    run_process(config_cmd)


def handle_args():
    parser = argparse.ArgumentParser(description="""Run HCL demo test""")

    parser.add_argument("--size", metavar="N", type=int, help="Data size is 2^N")
    parser.add_argument("--size_in_bytes", metavar="N", type=int, help="Data size is N")
    parser.add_argument(
        "--size_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Test will run on all powers of 2 from 2^MIN to 2^MAX (must satisfy: 1 <= MIN < MAX <=32)",
    )
    parser.add_argument(
        "--size_range_base",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Test will run on all sizes from 2^MIN*num_ranks to 2^MAX*num_ranks (must satisfy: 1 <= MIN < MAX <=32)",
    )
    parser.add_argument("--factor", metavar="N", type=int, help="Factor is N (for 'all2all_v' test)")
    parser.add_argument("--loop", type=int, help="Number of loop iterations")
    parser.add_argument(
        "--max_comm_id",
        metavar="COMM_ID",
        type=int,
        help="Max communicator ID from input JSON HCL_COMMUNICATORS section",
    )
    parser.add_argument("--json", default="hls1.json", type=str, help="Path to json config file (default is hls1.json)")
    parser.add_argument("--a2av_custom_file", type=str, help="Path to configuration file for all2all_v_custom test")
    parser.add_argument(
        "--data_type", metavar="TYPE", default="bfloat16", type=str, help="TYPE=<float|bfloat16> (default=bfloat16)"
    )
    parser.add_argument("--test", metavar="TEST", type=str, help="Specify test (use '-l' option for test list)")
    parser.add_argument("-l", "--list_tests", action="store_true", help="Show list of available tests")
    parser.add_argument("--same_address", action="store_true", help="Use same address for RDMA operations")
    parser.add_argument(
        "--no_weak_order",
        action="store_true",
        help="Do not use weak order. The networking of an operation cannot begin in parallel to the compute of the previous operation",
    )
    parser.add_argument("--no_correctness", action="store_true", help="Skip correctness validation")
    parser.add_argument("--no_performance", action="store_true", help="Skip performance measurement")
    parser.add_argument(
        "--perf_mode",
        action="store_true",
        help="Set scaling governor on all CPUs to performance mode. Not for use on VMs",
    )
    parser.add_argument(
        "--module_ids",
        type=int,
        nargs="*",
        choices=range(8),
        help="List of Module IDs to assign per test process. List size must match the # of HCL ranks in JSON config file.",
    )
    parser.add_argument("-g", "--gdb", type=int, default=-1, help="Run process with gdb")
    parser.add_argument("-console", action="store_true", help="Emit logs to terminal console (in addition to log file)")
    parser.add_argument(
        "-prof",
        type=str,
        nargs="*",
        choices=[str(i) for i in range(8)] + ["all"],
        help="Enable HW profiling per rank (e.g. '-prof 0 1 2'/'-prof all')",
    )
    parser.add_argument(
        "-clean",
        action="store_true",
        help="Clean previous artifacts including logs, recipe and csv results and exit if no test is specified",
    )
    parser.add_argument("-tee", action="store_true", help="Duplicate output to files: <i>.txt")
    parser.add_argument("-log", action="store_true", help="Redirect output to files: log<i>.txt")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging from this wrapper script")
    # System CPU settings
    parser.add_argument(
        "--num_cores_per_socket",
        metavar="CORES",
        type=int,
        default=28,
        help="Number of cores per socket - default is 18",
    )
    parser.add_argument(
        "--num_sockets", metavar="SOCKETS", type=int, default=2, help="Number of cpu sockets - default is 2"
    )
    parser.add_argument(
        "--num_hyper_threads",
        metavar="HYPER_THREADS",
        type=int,
        default=2,
        help="Number of hyper threads per core - default is 2",
    )
    parser.add_argument(
        "--proc_affinity",
        action="store_true",
        help="Enable binding num_hyper_threads cores per process (Supported for bare metal only)",
    )
    args = parser.parse_args()

    if args.verbose:
        global print_log
        print_log = lambda *x: print("[HCL_DEMO]", *x)
        print_log("Verbose printing enabled")

    if args.size:
        test_params["size"] = 1 << args.size

    if args.size_in_bytes:
        test_params["size"] = args.size_in_bytes

    if args.size_range:
        min_size_power = args.size_range[0]
        max_size_power = args.size_range[1]
        if not (1 <= min_size_power <= 31) or not (1 <= max_size_power <= 31) or max_size_power <= min_size_power:
            # parse error
            print(f"Invalid size range parameters: {min_size_power} {max_size_power}")
            sys.exit(1)
        test_params["size_range"] = args.size_range

    elif args.size_range_base:
        min_size_base = args.size_range_base[0]
        max_size_base = args.size_range_base[1]
        if not (1 <= min_size_base <= 30) or not (1 <= max_size_base <= 30) or max_size_base <= min_size_base:
            # parse error
            print(f"Invalid size range parameters: {min_size_base} {max_size_base}")
            sys.exit(1)
        test_params["size_range_base"] = args.size_range_base

    if args.factor:
        if args.test == "all2all_v":
            test_params["size"] = args.factor
        else:
            print("Error: Factor option is valid for 'all2all_v' test only")
            sys.exit(1)

    if args.loop:
        test_params["loop"] = args.loop

    if args.json:
        test_params["json_file"] = args.json

    if args.max_comm_id:
        test_params["max_comm_id"] = args.max_comm_id

    if args.a2av_custom_file:
        test_params["a2av_custom_file"] = args.a2av_custom_file

    if args.test:
        if not args.test in test_list:
            print(f"Error: no test {args.test}. Select a test from the list:")
            show_test_list()
            sys.exit(1)
        test_params["test"] = args.test

    if args.list_tests:
        show_test_list()
        sys.exit(0)

    if args.data_type:
        test_params["data_type"] = args.data_type

    if args.same_address:
        test_params["same_address"] = True

    if args.no_weak_order:
        test_params["no_weak_order"] = True

    if args.no_correctness:
        test_params["skip_correctness"] = True

    if args.no_performance:
        test_params["skip_performance"] = True

    if args.perf_mode:
        global perf_mode
        perf_mode = True

    if args.prof or "-prof" in sys.argv:
        configure_profiler()
        if args.prof:
            test_params["prof"] = args.prof
        else:
            test_params["prof"] = ["0"]

    if args.clean:
        clean_artifacts()
        if args.test == None:
            sys.exit(0)

    if args.module_ids:
        global module_ids
        module_ids = args.module_ids

    if args.gdb != None:
        test_params["gdb"] = args.gdb

    if args.console:
        test_params["enable_console"] = "true"

    if args.log:
        test_params["isLog"] = True

    if args.tee:
        test_params["isTee"] = True

    if args.num_cores_per_socket != None:
        test_params["num_cores_per_socket"] = args.num_cores_per_socket

    if args.num_sockets != None:
        test_params["num_sockets"] = args.num_sockets

    if args.num_hyper_threads != None:
        test_params["num_hyper_threads"] = args.num_hyper_threads

    if args.proc_affinity:
        test_params["proc_affinity"] = True


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def main():
    handle_args()
    print_log("Printing test params:")
    print_log(test_params)

    read_settings()

    # validate setting Module IDs list size match #of ranks
    if len(module_ids) != 0 and len(module_ids) != num_processes:
        print(f"Error: {num_processes} HCL Ranks configured with unmatched {len(module_ids)} module_ids arg")
        sys.exit(1)
    # Module ID from 'HLS_MODULE_ID' env var required to set for single process
    module_id = int(os.getenv("HLS_MODULE_ID", -1))
    if module_id != -1 and num_processes != 1:
        print(
            f"Error: setting 'ID' env var is valid only when run on one rank, for multiple ranks use --module_ids option"
        )
        sys.exit(1)

    # with one process override module_ids with ID env var if exist, default=0
    if num_processes == 1:
        if module_id == -1:
            if len(module_ids) == 0:
                module_id = 0
            else:
                module_id = module_ids[0]

    # create the test executable if not found
    if not os.path.exists(demo_exe):
        handle_make()

    if perf_mode and not "skip_performance" in test_params:
        perf_cmd = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        run_process(perf_cmd)

    # prepare and launch test processes
    processes = []

    # for single process use the ID env var to select card to run on
    if num_processes == 1:
        p = get_demo_cmd(id=module_id, **test_params)
        processes.append(p)
    # for multi process use module_ids arg or sequence id start from 0
    else:
        # if provided module_ids arg, use it
        if num_processes == len(module_ids):
            for i in range(num_processes):
                if i == debuggerID:
                    continue
                p = get_demo_cmd(id=module_ids[i], **test_params)
                processes.append(p)
        # for multi process use sequence id start from 0
        else:
            for i in range(num_processes):
                if i == debuggerID:
                    continue
                p = get_demo_cmd(id=i, **test_params)
                processes.append(p)

    # w = start_dl_workers()

    # pool = Pool(processes=num_processes)
    pool = MyPool(num_processes)
    results = pool.map(run_process, processes)
    # results = pool.imap_unordered(run_process, processes)

    for res in results:
        if res != 0:
            print("One of the hcl_test processes failed, terminating hcl demo.")
            pool.close()
            pool.terminate()
            pool.join()
            os.killpg(0, signal.SIGTERM)
            sys.exit(os.EX_DATAERR)
            break

    pool.close()
    pool.join()

    # stop_dl_workers(w)


if __name__ == "__main__":
    main()
