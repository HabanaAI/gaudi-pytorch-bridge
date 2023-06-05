import os
import argparse
import sqlite3
import subprocess

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

    # Check if the data in each table is the same in both databases
    differing_tables = {}
    for table_name in tables1[2:]:
        cursor1.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows1 = cursor1.fetchall()
        cursor2.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows2 = cursor2.fetchall()
        if len(rows1) != len(rows2):
            print(table_name[0] + " has different tensor numbers in static and dynamic not comparing")
        else :
            row1_data = [lis[-1] for lis in rows1]
            row2_data = [lis[-1] for lis in rows2]
            if row1_data != row2_data:
                differing_tables[table_name] = []
                #TODO : Add tolerance for comparision
                # may need to use db_parser to serialize the data and compare
                # perhaps use db_parser find maximum difference and dump in csv
                for i in range(len(rows1)):
                    if(rows1[i][-1] != rows2[i][-1]):
                       differing_tables[table_name].append(rows1[i][4])

    if differing_tables:
        print("The following tables have different data:")
        for key, value in differing_tables.items():
            print(key, value)
        return False
    else:
        return True


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

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cmd', help='the workload command to dump and compare static and dynamic runs tensor data ', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    dump_dir = '/tmp'
    SYNREC_PATH=get_synrec_path()
    # Loop through all files in the current directory
    for file in os.listdir(dump_dir):
        if file.endswith(".json") or file.endswith(".db") or file.endswith(".lock"):
            print('Deleting file ' + dump_dir + '/' + file)
            os.remove(os.path.join(dump_dir, file))
    synrec_command_static = f'PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1 PT_HPU_ENABLE_MIN_MAX_AS_CURRENT=1 {SYNREC_PATH} -t -p /tmp/StaticSynRec --ignore-errors --overwrite -- {args.cmd}'
    synrec_command_dynamic = f'PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1 {SYNREC_PATH} -t -p /tmp/DynamicSynRec --ignore-errors --overwrite -- {args.cmd}'
    script_attr = "script -q -c"
    for cmd in [synrec_command_static, synrec_command_dynamic]:
        print(cmd)
        process = subprocess.Popen(script_attr + '"' + cmd + '"', stdout=subprocess.PIPE, shell=True)
        while True:
            output = process.stdout.readline().decode()
            if not output:
                break
            print(output, end='')
    static_path = '/tmp/StaticSynRec.db'
    dynmaic_path = '/tmp/DynamicSynRec.db'
    if os.path.getsize(static_path) == 0 or os.path.getsize(dynmaic_path) == 0:
        print("The synrec didn't run correctly, db files are empty")
    else:
        is_equal =  compare_databases(static_path, dynmaic_path)
        if is_equal:
            print("The static and dynamic runs are equal")
        else:
            print("There is divergence in static and dynamic runs")
