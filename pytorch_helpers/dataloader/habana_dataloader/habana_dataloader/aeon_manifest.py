import os
import random
import tempfile


def _is_valid_image(image):
    return image.lower().endswith(".jpeg") or image.lower().endswith(".jpg")


def generate_aeon_manifest(imgs):
    """Generates aeon manifest file from ImageFolder dataset
    Args:
        imgs: a list of Tuples: (file_path, label_num)
    """
    manifest = tempfile.NamedTemporaryFile(mode="w", delete=False)
    manifest.write("@FILE\tSTRING\n")

    for file_path, label_num in imgs:
        if not _is_valid_image(file_path):
            raise ValueError(
                "HabanaDataLoader supports only jpg/jpeg files, found unsupported file: {}.".format(file_path)
            )
        manifest.write(str(file_path) + "\t" + str(label_num) + "\n")
    manifest.close()
    return manifest.name
