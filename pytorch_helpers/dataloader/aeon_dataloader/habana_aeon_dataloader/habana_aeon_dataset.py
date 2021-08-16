# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from torch.utils.data import IterableDataset
import habana_torch_dataloader
import habana_aeon_dataloader.aeon_app
from .aeon_config import get_aeon_config
from pathlib import Path
import json


class HabanaAeonIterableDataset(IterableDataset):
    def __init__(self, aeon_data_dir, manifest_filename, batch_size, height, width, instance_id, num_instances, is_train=True):
        aeon_config_json = get_aeon_config(aeon_data_dir, manifest_filename, batch_size, workers, height, width, is_train)
        self.aeon = habana_aeon_dataloader.aeon_app.AeonPytorchDL(aeon_config_json, True, True)

    def __iter__(self):
        return iter(self.aeon)

    def __len__(self):
        return len(self.aeon)

    def __next__(self):
        return next(self.aeon)

# Aeon backend does batch collation, hence Dataloader should always return False
class AeonDataLoader(habana_torch_dataloader.DataLoader):
    @property
    def _auto_collation(self):
        return False
