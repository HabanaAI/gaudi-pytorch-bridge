import os
import tempfile
import random

def generate_aeon_manifest(data_directory: str):
    classes = sorted(entry.name for entry in os.scandir(data_directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {data_directory}.")

    manifest = tempfile.NamedTemporaryFile(mode='w', delete=False)

    manifest.write("@FILE\tSTRING\n")
    for label_num, class_name in enumerate(classes):
        class_full_path = os.path.join(data_directory, class_name)
        for image in os.listdir(class_full_path):
            if not image.endswith('.JPEG') and not image.endswith('.jpeg'):
                continue
            file_path = os.path.join(class_name, image)
            manifest.write(str(file_path) + "\t" + str(label_num) + "\n")

    manifest.close()
    return manifest.name
