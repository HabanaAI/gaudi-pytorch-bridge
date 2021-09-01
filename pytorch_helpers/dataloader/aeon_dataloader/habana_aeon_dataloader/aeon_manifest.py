import os

AEON_MANIFEST_PATH = 'aeon_manifest.txt'

def generate_aeon_manifest(data_directory: str, output: str = AEON_MANIFEST_PATH):
    classes = sorted(entry.name for entry in os.scandir(data_directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {data_directory}.")

    with open(output, 'w') as manifest:
        for label_num, class_name in enumerate(classes):
            class_full_path = os.path.join(data_directory, class_name)
            for image in os.listdir(class_full_path):
                file_path = os.path.join(class_name, image)
                manifest.write(str(file_path) + "\t" + str(label_num) + "\n")

    return os.path.abspath(output)
