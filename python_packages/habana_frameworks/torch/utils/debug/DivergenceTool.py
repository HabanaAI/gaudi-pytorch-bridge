import os
import argparse
import sqlite3
import subprocess
import csv
import numpy as np
import json

dump_dir = '/tmp/'
static_path = dump_dir + 'StaticSynRec.db'
dynamic_path = dump_dir + 'DynamicSynRec.db'
static_json = dump_dir + 'StaticSynRec.json'
dynamic_json = dump_dir + 'DynamicSynRec.json'
differing_tables = {}

def ca_tensor_error_stats(a,b):
    d = np.subtract(a,b)
    ad = np.abs(d)
    maxabs = np.amax(ad)
    minabs = np.amin(ad)

    # h,_ = np.histogram(ad,5)
    # n = ad.size
    # h = np.divide(h,n)
    # h = np.multiply(h,100.0)
    # h = np.around(h,2)

    dsq = np.square(d)
    mse = np.mean(dsq)
    rmse = np.sqrt(mse)
    return maxabs.item(), minabs.item(),mse.item(), rmse.item()

def ca_cosine_similarity(a, b, cos_sim_thld, rms_threshold):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if (na.item() == 0.0) or (nb.item() == 0.0):
        l2_norm = np.linalg.norm(np.array(a)-np.array(b))
        cos_sim_ok = np.greater(rms_threshold, l2_norm/np.sqrt(a.size))
        angle = 0 if cos_sim_ok else 90
        nr = 1.0 if cos_sim_ok else 100
        return na.item(),nb.item(),nr, angle, cos_sim_ok
    else:
        nr =  np.divide(na,nb)
        angle = np.arccos(min(np.dot(a, b) / na / nb, 1.0))/np.pi*180
        angle = np.around(angle,2)
        cos_sim_ok =  np.greater(cos_sim_thld , angle) or np.greater(rms_threshold,na/np.sqrt(a.size))
        if np.greater(angle,cos_sim_thld) and cos_sim_ok:
            angle = np.float32(0.99)
            print(f'Cosine similarity marked True as RMS was below threshold, calculated angle is {angle.item()} set to 0.99')
        return na.item(),nb.item(),nr.item(), angle.item(), cos_sim_ok

def calculate_difference(static_val, dynamic_val):
    np_static_val = np.array(static_val).astype(np.float64)
    np_dynamic_val = np.array(dynamic_val).astype(np.float64)
    # 1 degree threshold for cosine ca_cosine_similarity
    cos_sim_thld = 1.0
    rms_threshold=1e-10
    maxabs, minabs, mse, rmse = ca_tensor_error_stats(np_static_val, np_dynamic_val)
    norm_dev1, norm_dev2, norm_r, angle, cos_sim_ok = ca_cosine_similarity(np_static_val, np_dynamic_val, cos_sim_thld, rms_threshold)
    return [maxabs, minabs, mse, rmse, norm_dev1, norm_dev2, norm_r, angle, cos_sim_ok]


def write_tuples_to_csv(data, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def run_commands(command_static, command_dynamic, no_print=False):
    script_attr = "script -q -c"
    for cmd in [command_static, command_dynamic]:
        if not no_print:
            print(cmd)
        process = subprocess.Popen(script_attr + '"' + cmd + '"', stdout=subprocess.PIPE, shell=True)
        while True:
            output = process.stdout.readline().decode()
            if not output:
                break
            if not no_print:
                print(output, end='')

# Function to read values from a file
def read_values_from_file(file_name):
    values = []
    with open(file_name, 'r') as file:
        # skip first line as it contains tensor name
        next(file)
        for line in file:
            value = float(line.strip())
            values.append(value)
    return values

def get_synrec_path():
    # specify the possible file paths
    file_path = None
    synapse_root = os.environ.get('SYNAPSE_ROOT')
    if synapse_root is not None :
        file_path = os.path.join(synapse_root, "scripts", "synrec.py")
    possible_paths = [
        "/root/repos/synapse/scripts/synrec.py"
        "/root/synapse/scripts/synrec.py",
        file_path
    ]

    # check if any of the file paths exist
    for path in possible_paths:
        if os.path.isfile(path):
            return path
    else:
        print("Could not find synrec.py file in either path")
        return None

def get_json_test_path():
    # specify the possible file paths
    file_path = None
    synapse_root = os.environ.get('SYNAPSE_RELEASE_BUILD')
    if synapse_root is not None :
        file_path = os.path.join(synapse_root, "bin", "json_tests")
    possible_paths = [
        file_path
    ]
    # check if any of the file paths exist
    for path in possible_paths:
        if os.path.isfile(path):
            return path
        else:
            print("Could not find json_tests file")
            return None

def compare_databases(db_file1, db_file2):
    conn1 = sqlite3.connect(db_file1)
    conn2 = sqlite3.connect(db_file2)

    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()

    # Get a list of tables in both databases
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables1 = cursor1.fetchall()
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables2 = cursor2.fetchall()

    # Check if the number of tables is the same in both databases
    if len(tables1) != len(tables2):
        print("Databases have different number of tables")
        return False

    # Check if the tables have the same names in both databases
    if sorted(tables1) != sorted(tables2):
        print("Databases have different table names")
        return False
    '''
    This is thr format in which data is preset in DB file
    Data is from synapse/src/data_serialize/sql_db_serializer.cpp
        "ROW_INDEX      int     not NULL,"
        "ID             int     not NULL,"
        "GRAPH_GROUP    int     not NULL,"
        "ITERATION      int     not NULL,"
        "NAME           text    not NULL,"
        "TYPE           int     not NULL,"
        "DATA_TYPE      int     not NULL,"
        "COMPRESSION    int     not NULL,"
        "VALIDATION     int     not NULL," -> If this is valid tensor(0->valid)
        "CONST_TENSOR   int     not NULL,"
        "SHAPE          blob,"
        "PERMUTATION    blob,"
        "DATA_ID        int     not NULL," -> Hash of the data present in tensor
    '''
    valid_idx =  8
    name_idx = 4
    data_idx = -1
    # Check if the data in each table is the same in both databases
    for table_name in tables1[2:]:
        cursor1.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows1 = cursor1.fetchall()
        cursor2.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows2 = cursor2.fetchall()
        if len(rows1) != len(rows2):
            print(table_name[0] + " has different tensor numbers in static and dynamic not comparing")
        else:
            row1_data = [lis[data_idx] for lis in rows1]
            row2_data = [lis[data_idx] for lis in rows2]
            if row1_data != row2_data:
                differing_tables[table_name] = []
                for i in range(len(rows1)):
                    # Check if tensor is valid and data is different
                    if((rows1[i][valid_idx] == 0) and (rows2[i][valid_idx] == 0 ) and (rows1[i][data_idx] != rows2[i][data_idx])):
                       differing_tables[table_name].append(rows1[i][name_idx])

    if differing_tables:
        print("There is divergence in static and dynamic runs")
        print("Dumping divergence data in file\033[91m DivergenceDump.txt\033[0m")
        output_file = open('DifferenceDump.txt', 'w')
        for key, value in differing_tables.items():
            print(key, value, file=output_file)
        output_file.close()
        return False
    else:
        print("The static and dynamic runs are equal")
        return True

def find_node_name(graph_name, tensor):
    with open(dynamic_json, "r") as file:
        json_data = json.load(file)
        for graph in json_data["graphs"]:
            if graph["name"] == graph_name:
                for node in graph["nodes"]:
                    if tensor in node["output_tensors"]:
                        return ['Output', node["name"], node["guid"]]
                    elif tensor in node["input_tensors"]:
                        return ['Input', node["name"], node["guid"]]
    return ['None', 'None', 'None']

def dump_to_csv():
    print("---------- Analyzing differences using dbparser and dumping in CSV file \033[91msynrec_comparision.csv\033[0m ----------")
    dynamic_output = 'Output_dynamic.log'
    static_output = 'Output_static.log'
    json_tests_path = get_json_test_path()
    csv_dump_list = []
    csv_tuple = ('Graph', 'Tensor', 'Input/Output', 'Node Name', 'Node Guid', 'maxabs', 'minabs', 'mse', 'rmse', 'norm_dev1', 'norm_dev2', 'norm_r', 'angle', 'cos_sim_ok')
    csv_dump_list.append(csv_tuple)
    for key, values in differing_tables.items():
        for value in values:
            print("Analyzing graph:", key[0], "-> Tensor:", value)
            command_dump_static = f"{json_tests_path} db_parser -d {static_path} -g '{key[0]}' -t '{value}' -o {static_output}"
            command_dump_dynamic = f"{json_tests_path} db_parser -d {dynamic_path} -g '{key[0]}' -t '{value}' -o {dynamic_output}"
            if os.path.isfile(static_output):
                os.remove(static_output)
            if os.path.isfile(dynamic_output):
                os.remove(dynamic_output)
            run_commands(command_dump_static, command_dump_dynamic, True)
            if not(os.path.isfile(dynamic_output)) or not (os.path.isfile(static_output)):
                print('\033[91m'+ "skip dumping ", value, 'to CSV \033[0m')
                continue
            static_values = read_values_from_file(static_output)
            dynamic_values = read_values_from_file(dynamic_output)
            # return format [maxabs, minabs, mse, dist, rmse, norm_dev1, norm_dev2, norm_r, angle, cos_sim_ok]
            difference = calculate_difference(static_values, dynamic_values)
            node_name = find_node_name(key[0], value)
            csv_tuple = (key[0], value, node_name[0], node_name[1], node_name[2])
            combined_tuple = csv_tuple + tuple(difference)
            csv_dump_list.append(combined_tuple)
    if os.path.isfile(static_output):
        os.remove(static_output)
    if os.path.isfile(dynamic_output):
        os.remove(dynamic_output)
    write_tuples_to_csv(csv_dump_list, "synrec_comparision.csv")

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cmd', help='the workload command to dump and compare static and dynamic runs tensor data ', required=True)
parser.add_argument("--csv", action="store_true", help="Specify if difference should be in CSV format")
args = parser.parse_args()

if __name__ == '__main__':
    SYNREC_PATH=get_synrec_path()
    # Loop through all files in the current directory
    for file in os.listdir(dump_dir):
        if file.endswith(".json") or file.endswith(".db") or file.endswith(".lock"):
            print('Deleting file ' + dump_dir + '/' + file)
            os.remove(os.path.join(dump_dir, file))
    default_config = "PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1"
    synrec_command_static = f'{default_config} PT_HPU_ENABLE_MIN_MAX_AS_CURRENT=1 {SYNREC_PATH} -t -p /tmp/StaticSynRec --ignore-errors --overwrite -- {args.cmd}'
    synrec_command_dynamic = f'{default_config} {SYNREC_PATH} -t -p /tmp/DynamicSynRec --ignore-errors --overwrite -- {args.cmd}'
    run_commands(synrec_command_static, synrec_command_dynamic)
    if os.path.getsize(static_path) == 0 or os.path.getsize(dynamic_path) == 0:
        print("The synrec didn't run correctly, db files are empty")
    else:
        compare_databases(static_path, dynamic_path)
    if args.csv:
        dump_to_csv()
