import os
import argparse
import sqlite3

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
    differing_tables = []
    for table_name in tables1:
        cursor1.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows1 = cursor1.fetchall()
        cursor2.execute(f"SELECT * FROM \"{table_name[0]}\"")
        rows2 = cursor2.fetchall()

        if rows1 != rows2:
            # Get the primary key of the table
            cursor1.execute(f"PRAGMA table_info(\"{table_name[0]}\")")
            table_info = cursor1.fetchall()
            pk_column = [info[1] for info in table_info if info[5] == 1][0]
            differing_tables.append((table_name[0], pk_column))

    if differing_tables:
        print("The following tables have different data:")
        for table in differing_tables:
            print(f"{table[0]} (primary key: {table[1]})")
            print(f"Data in {db_file1}:")
            cursor1.execute(f"SELECT * FROM \"{table[0]}\"")
            rows1 = cursor1.fetchall()
            for row in rows1:
                print(row)
            print(f"Data in {db_file2}:")
            cursor2.execute(f"SELECT * FROM \"{table[0]}\"")
            rows2 = cursor2.fetchall()
            for row in rows2:
                print(row)
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
    current_dir = os.getcwd()
    SYNREC_PATH=get_synrec_path()
    # Loop through all files in the current directory
    for file in os.listdir(current_dir):
        if file.endswith(".json") or file.endswith(".db") or file.endswith(".lock"):
            print('Deleting file ' + file)
            os.remove(os.path.join(current_dir, file))
    synrec_command_static = f'PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1 PT_HPU_ENABLE_MIN_MAX_AS_CURRENT=1 {SYNREC_PATH} -t -p ./StaticSynRec --ignore-errors --overwrite -- {args.cmd}'
    synrec_command_dynamic = f'PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_PGM_ENABLE_CACHE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1 {SYNREC_PATH} -t -p ./DynamicSynRec --ignore-errors --overwrite -- {args.cmd}'
    script_attr = "script -q -c"
    for cmd in [synrec_command_static, synrec_command_dynamic]:
        print(cmd)
        process = os.popen(script_attr + '"' + cmd + '"')
        output = process.read()

    is_equal =  compare_databases('./StaticSynRec.db', './DynamicSynRec.db')
    if is_equal:
        print("The static and dynamic runs are equal")
    else:
        print("There is divergence in static and dynamic runs")
