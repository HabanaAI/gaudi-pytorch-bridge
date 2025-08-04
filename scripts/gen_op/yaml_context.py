###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from typing import Any

import yaml


class YamlContext:
    def __init__(self, op_data, template_map):
        self.template_map = template_map
        self.op_data = self.parse_yaml(op_data)

    def get_op_names(self):
        return self.op_data.keys()

    def get_op_data(self):
        return self.op_data.items()

    def parse_yaml(self, yaml_data: dict[str, Any]) -> dict[str, Any]:
        for op, fields in list(yaml_data.items()):
            if "op_templates" in fields:
                yaml_data[op] = self.apply_templates(fields)

        return yaml_data

    def apply_templates(self, op_fields: dict[str, Any]) -> dict[str, Any]:
        """
        Apply templates to the op fields. We allow multi-level and multiple templates.
        An only condition is that for templates having common field and being on common level
        we require that this field will be overwritter on higher level."""
        nested = {}
        fields_in_templates = []
        for template_name in op_fields.get("op_templates", []):
            fields_in_templates.append(set(self.template_map[template_name].keys()))

            template_fields = self.apply_templates(self.template_map[template_name])
            nested.update(template_fields)

        merged = {k: v for k, v in op_fields.items() if k != "op_templates"}
        self.check_for_duplicate_keys(fields_in_templates, merged)

        merged = nested | merged

        return merged

    def check_for_duplicate_keys(self, fields_in_templates, merged) -> None:
        all_keys = set()
        duplicated_keys = set()
        for fields in fields_in_templates:
            duplicated_keys |= all_keys.intersection(fields)
            all_keys |= fields

        for key in duplicated_keys:
            assert key in merged.keys(), (
                f"For fields that occurs in multiple templates require field: {key} to be defined explicitly."
            )


def yaml_context_from_files(yaml_file, template_file):
    with open(yaml_file) as ff:
        op_data = yaml.load(ff.read(), Loader=yaml.CSafeLoader)

    with open(template_file) as tf:
        template_map = yaml.load(tf.read(), Loader=yaml.CSafeLoader)

    return YamlContext(op_data, template_map)
