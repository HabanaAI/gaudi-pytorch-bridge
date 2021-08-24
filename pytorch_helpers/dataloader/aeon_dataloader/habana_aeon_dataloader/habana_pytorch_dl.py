# TODO: this file should be deleted in the next AEON DL commit

# Must import torch before loading torch-cpp extension
import torch
import habana_aeon_dataloader.aeon_app

IMG_HEIGHT = 224
IMG_WIDTH = 224

def getAeonConfig(aeon_data_dir, manifest_file, batch_size, instance_id, num_instances):
  return {
      "augmentation": [
          {
          "caffe_mode": True,
          "center": False,
          "crop_enable": True,
          "do_area_scale": True,
          "flip_enable": True,
          "horizontal_distortion": [
              0.75,
              1.33333337306976
          ],
          "scale": [
              0.08,
              1.0
          ],
          "type": "image"
          }
      ],
      "batch_size": batch_size,
      "decode_thread_count": 8,
      "fread_thread_count": 4,
      "instance_id": instance_id,
      "num_instances": num_instances,
      "random_seed": 1,
      "file_shuffle_seed": 5,
      "shuffle_manifest": True,
      "etl": [
          {
          "type": "image",
          "channel_major": False,
          "height": IMG_HEIGHT,
          "width": IMG_WIDTH,
          "output_type": "float"
          },
          {
          "binary": False,
          "type": "label"
          }
      ],
      "iteration_mode": "ONCE",
      "manifest_filename": manifest_file,
      "manifest_root": aeon_data_dir
      }

class HabanaTorchDL(habana_aeon_dataloader.aeon_app.AeonPytorchDL):
    def __init__(self, aeon_data_dir, manifest_file, batch_size, instance_id, num_instances):
        super().__init__(getAeonConfig(aeon_data_dir, manifest_file, batch_size, instance_id, num_instances), True, True)
