###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import copy
import os
import sys

import pytest
from xfail_hpu import all_hangs_dict, all_xfails_dict, import_error_files_to_ignore

env_flags_backup = {}
from cpu_failed_tests import all_cpu_failed_dict
from hpu_failed_cpu_fallback_disabled import all_hpu_failed_cpu_fallback_disabled_dict
from skip_hpu import all_skipped_dict

# Ignore all the test files which throw python import errors
collect_ignore = copy.deepcopy(import_error_files_to_ignore)
regressions_to_ignore = []


# Key in the expected_fail_tests can be an exact node_id or module or directory(Ex: tensorflow/python/keras/distribute)
def node_in_test_dict(test_dict, nodeid):
    for key in test_dict:
        if key.endswith("::") or key.endswith(".py") or key.endswith("/"):
            if nodeid.startswith(key):
                return True
        elif key == nodeid:
            return True

    return False


def get_reason_for_node(test_dict, nodeid):
    for key in test_dict:
        if key.endswith("::") or key.endswith(".py") or key.endswith("/"):
            if nodeid.startswith(key):
                return test_dict[key]
        elif key == nodeid:
            return test_dict[key]

    return ""


def pytest_addoption(parser):
    parser.addoption("--without_hpu", action="store_true", default=False, help="Disable HPU and run tests")
    parser.addoption("--enable_graph_dumps", action="store_true", default=False, help="Enable graph dumps")
    parser.addoption(
        "--run_xfail_only", action="store_true", default=False, help="Runs all the xfail marked tests alone"
    )
    parser.addoption(
        "--run_hang_tests", action="store_true", default=False, help="Runs only the tests which causes hang/crash"
    )
    parser.addoption(
        "--run_jira",
        action="store",
        default="",
        help="Runs the failed tests that are marked under this jira(--run_jira SW-xxxx). To run all the jiras: --run_jira all",
    )


def pytest_sessionstart(session):
    without_hpu = session.config.getoption("--without_hpu")
    if without_hpu:
        return

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    import glob

    for file in glob.glob(cur_dir + "/*.regressions"):
        with open(file, "r") as fp:
            regressions_to_ignore.extend(fp.read().splitlines())


def pytest_collection_modifyitems(config, items):
    # TODO: to add xfail options in run_habana.py
    run_xfail_only = config.getoption("--run_xfail_only")
    run_hang_tests_only = False  # config.getoption("--run_hang_tests")
    jira_id = config.getoption("--run_jira")

    deselected_list = []
    if jira_id:
        tests_to_run = []

        if jira_id == "all":
            jira_id = "https://jira.habana-labs.com/browse/"
        else:
            jira_id = "https://jira.habana-labs.com/browse/" + jira_id

        all_fail_tests = {**all_xfails_dict}

        for item in items:
            reason_str = get_reason_for_node(all_fail_tests, item.nodeid)
            if node_in_test_dict(all_fail_tests, item.nodeid) and reason_str.startswith(jira_id):
                xfail_marker = pytest.mark.xfail(run=True, reason=reason_str)
                item.add_marker(xfail_marker)
                tests_to_run.append(item)
            else:
                deselected_list.append(item)

        items[:] = tests_to_run
        config.hook.pytest_deselected(items=deselected_list)
    else:
        xfail_test_list = []
        hang_test_list = []

        for item in items:
            timeout_marker = pytest.mark.timeout(timeout=300, method="signal")
            item.add_marker(timeout_marker)
            if node_in_test_dict(all_xfails_dict, item.nodeid):
                reason_str = get_reason_for_node(all_xfails_dict, item.nodeid)
                xfail_marker = pytest.mark.xfail(run=run_xfail_only, reason=reason_str)
                xfail_test_list.append(item)
                item.user_properties.append(("xfail", "true"))
                item.add_marker(xfail_marker)
            elif node_in_test_dict(all_hangs_dict, item.nodeid):
                reason_str = get_reason_for_node(all_hangs_dict, item.nodeid)
                xfail_marker = pytest.mark.xfail(run=run_hang_tests_only, reason=reason_str)
                hang_test_list.append(item)
                item.user_properties.append(("xfail", "true"))
                item.add_marker(xfail_marker)
            elif node_in_test_dict(all_hpu_failed_cpu_fallback_disabled_dict, item.nodeid):
                reason_str = get_reason_for_node(all_hpu_failed_cpu_fallback_disabled_dict, item.nodeid)
                xfail_marker = pytest.mark.xfail(run=run_xfail_only, reason=reason_str)
                xfail_test_list.append(item)
                item.user_properties.append(("xfail", "true"))
                item.add_marker(xfail_marker)
            elif item.nodeid in regressions_to_ignore:
                reason_str = "Regressions from the current build"
                xfail_marker = pytest.mark.xfail(run=run_xfail_only, reason=reason_str)
                xfail_test_list.append(item)
                item.add_marker(xfail_marker)
            else:
                deselected_list.append(item)
            # skip takes precedence incase there are overalapping
            if "_hpu" in item.nodeid and "test_contig_size1_large_dim" in item.nodeid:
                skip_marker = pytest.mark.skip(reason="N-dim not supported on HPU")
                item.add_marker(skip_marker)
            if "_hpu" in item.nodeid and "test_noncontiguous_samples" in item.nodeid:
                skip_marker = pytest.mark.skip(reason="Storage access is not supported on HPU")
                item.add_marker(skip_marker)
            if "_hpu" in item.nodeid and "_complex" in item.nodeid:
                skip_marker = pytest.mark.skip(reason="Complex dtype is not supported on HPU")
                item.add_marker(skip_marker)
            if "_hpu" in item.nodeid and "_float64" in item.nodeid:
                skip_marker = pytest.mark.skip(reason="Float64 dtype is not supported on HPU")
                item.add_marker(skip_marker)
            if node_in_test_dict(all_skipped_dict, item.nodeid):
                reason_str = get_reason_for_node(all_skipped_dict, item.nodeid)
                skip_marker = pytest.mark.skip(reason_str)
                item.add_marker(skip_marker)
            if node_in_test_dict(all_cpu_failed_dict, item.nodeid):
                reason_str = get_reason_for_node(all_cpu_failed_dict, item.nodeid)
                skip_marker = pytest.mark.skip(reason=reason_str)
                item.add_marker(skip_marker)

        if run_xfail_only or run_hang_tests_only:
            items.clear()
            items[:] = xfail_test_list + hang_test_list
            config.hook.pytest_deselected(items=deselected_list)
