#!/usr/bin/env python3
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

# Utility script to parse all the Pytorch frameworktest report pytest xml files and generate a single csv file
# showing the statistics corresponding to each the xml files.
#
# Example generated csv file as below:
# PT FW Tests,Total Tests,Pass Tests,Error Tests,Failed Tests,Skipped Tests,Duration
#

import os
import sys
import xml.etree.ElementTree as ET


def get_xml_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith("." + extension))


def get_pass_percentage(pass_tests, total_tests, skipped_tests):
    if total_tests - skipped_tests == 0:
        return 100.0
    else:
        return round((pass_tests / (total_tests - skipped_tests)) * 100.0, 1)


def parse_xml_file(directory, xml_f):
    # xfail_expected_str to search in the xml files to calculate count of the xfailed tests
    xfail_expected_str = '<skipped type="pytest.xfail"'
    xfail_tests = 0

    error_tests = 0
    failed_tests = 0
    # Tests marked as xfail are counted in skips in the pytest xml file
    # so we will subtract the xfail tests count from the skipped tests count.
    skipped_tests = 0
    total_tests = 0
    # pass_tests will be (total_tests - (skip_count + failures_count + errors_count))
    pass_tests = 0
    duration = 0

    # Get the xfail count by searching for xfail_expected_str
    with open(directory + xml_f, "r") as file:
        xfail_tests = file.read().count(xfail_expected_str)

    # https://docs.python.org/3/library/xml.etree.elementtree.html
    # Below execution will get below information (as an example) in a dict
    # <testsuite errors="8" failures="1" name="pytest" skips="218" tests="1243" time="2552.161">
    root = ET.parse(directory + xml_f).getroot()
    testsuite_list = []
    if root.tag == "testsuite":
        testsuite_list.append(root.attrib)
    else:
        for child in root:
            if child.tag == "testsuite":
                testsuite_list.append(child.attrib)

    for testsuite_dict in testsuite_list:
        if "errors" in testsuite_dict:
            error_tests += int(testsuite_dict["errors"])

        if "failures" in testsuite_dict:
            failed_tests += int(testsuite_dict["failures"])

        # Some pytest reports have 'skips' and some have 'skipped' depending upon pytest versions
        if "skips" in testsuite_dict:
            skipped_tests += int(testsuite_dict["skips"])

        if "skipped" in testsuite_dict:
            skipped_tests += int(testsuite_dict["skipped"])

        if "tests" in testsuite_dict:
            total_tests += int(testsuite_dict["tests"])

        if "time" in testsuite_dict:
            duration += float(testsuite_dict["time"])

    skipped_tests -= xfail_tests
    pass_tests = total_tests - (skipped_tests + xfail_tests + failed_tests + error_tests)

    pass_percentage = get_pass_percentage(pass_tests, total_tests, skipped_tests)
    return [total_tests, pass_tests, error_tests, xfail_tests, failed_tests, skipped_tests, duration, pass_percentage]

    # return f"{total_tests},{pass_tests},{error_tests},{xfail_tests},{failed_tests},{skipped_tests},{duration},{pass_percentage}"


if __name__ == "__main__":

    # Path to the xml files is passed as an argument
    try:
        pytorch_framework_xml_dir = sys.argv[1]
    except:
        pytorch_framework_xml_dir = os.getcwd()

    pytorch_framework_xml_dir += "/"

    # Get all the xml files in a list
    pytorch_framework_xml_files = get_xml_files(pytorch_framework_xml_dir, "xml")

    # Set the csv file name to be generated
    pytorch_framework_csv_file = os.getcwd() + "/pytorch_framework_test_report.csv"

    aggregate_data = []
    for i in range(8):
        aggregate_data.append(0)

    # Create/truncate pytorch_framework_csv_file and add the header information
    with open(pytorch_framework_csv_file, "w") as file:
        file.write(
            "PT FW Test,Total Tests,Pass Tests,Error Tests,xFailed Tests,Failed Tests,Skipped Tests,Duration,Pass %\n"
        )

    # Parsing loop for the xml files
    for xml_file in sorted(pytorch_framework_xml_files):
        pytorch_test = xml_file.rpartition(".")[0]

        stats = parse_xml_file(pytorch_framework_xml_dir, xml_file)
        stats_str = "{},{},{},{},{},{},{},{}".format(*stats)
        for index, data in enumerate(stats):
            aggregate_data[index] = aggregate_data[index] + data

        with open(pytorch_framework_csv_file, "a") as file:
            # Appending statistics to the file
            file.write(f"{pytorch_test},{stats_str}\n")

    aggregate_pp = get_pass_percentage(aggregate_data[1], aggregate_data[0], aggregate_data[5])
    aggregate_data_str = "{},{},{},{},{},{},{}".format(*aggregate_data)
    with open(pytorch_framework_csv_file, "a") as file:
        file.write(f"Total,{aggregate_data_str},{aggregate_pp}\n")
