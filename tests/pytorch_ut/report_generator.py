###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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


import argparse
import csv
import enum
import logging
import os
import re
import sys
import traceback

from junitparser import Error, Failure, JUnitXml, Skipped


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_format(color, text):
    print(f"{color} {text} {bcolors.ENDC}")


logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


class SummaryLevel(enum.IntEnum):
    SUMMARY = (1,)
    FAIL = (2,)
    XFAIL = (4,)
    SKIP = (8,)


class Summary:
    module_name = ""
    num_pass = 0
    num_fail = 0
    num_error = 0
    num_xfail = 0
    num_skip = 0
    duration = 0

    def print(self):
        print(f"module: {self.module_name}")
        print(f"pass: {self.num_pass}")
        print(f"fail: {self.num_fail}")
        print(f"skip: {self.num_skip}")
        print(f"xfail: {self.num_xfail}")
        print(f"error: {self.num_error}")
        print(f"total: {self.total()}")
        print(f"duration: {self.duration}")

    def total(self):
        return self.num_pass + self.num_fail + self.num_xfail + self.num_error + self.num_skip

    def to_list(self):
        return [
            self.module_name,
            self.total(),
            self.num_pass,
            self.num_error,
            self.num_xfail,
            self.num_fail,
            self.num_skip,
            self.duration,
        ]


class TestModule:
    def __init__(self, name, xml, stdout):
        self.name = name
        self.xml = xml
        self.stdout = stdout

    def print(self):
        print("Module: ", self.name)
        print("\txml: ", self.xml)
        print("\tstdout: ", self.stdout)


class TestResult:
    def __init__(self, name, test_class, description, stack, stdout, stderr):
        self.name = name
        self.test_class = test_class
        self.description = description
        self.stack = stack
        self.stdout = stdout
        self.stderr = stderr

    def print(self):
        print("Test: ", self.name)
        print("test_class:", self.test_class)
        print("description:\n", self.description)
        print("stdout:\n", self.stdout)
        print("stderr:\n", self.stderr)
        print("stack trace:\n", self.stack)


class DictListValues:
    frequency = 0
    module = []

    def __init__(self, module, frequency):
        self.module = module
        self.frequency = frequency

    def __lt__(self, other):
        return self.frequency < other.frequency


# Error strings to get type of error
error_strings = [
    "RuntimeError",
    "AssertionError",
    "TypeError",
    "IndexError",
    "ValueError",
    "NotImplementedError",
    "GradcheckError",
]

# Regex rules to get error description
regex_rules = [
    ("Internal Error: Received signal - (.*)"),
    (r"(Exception raised .*) at (.*\.cpp:\d+)\s+(.*)"),
    ("([malloc][free])(.*)"),
    (r"(AssertionError:.*)!With.*(difference\(s\) exceeded the margin of error).*"),
    ("(.*Graph compile failed.)(.*)"),
    (".*GradcheckError: Jacobian mismatch(.*)"),
    ("(.*)While(.*)got(.*)"),
    ("(FCD stride > 1 is not supported.*)"),
    ("(AssertionError: False is not true : (.*) failed to compare as equal)(! Comparing.*)"),
]

# Get fail desc


def get_fail_desc(failure_type, message):
    for regex in regex_rules:
        desc = re.search(regex, str(message))
        if desc is None:
            continue
        if regex is regex_rules[0] and failure_type == "error":
            return message
        if regex is regex_rules[1] and failure_type == "error":
            return desc.group(1)
        if regex is regex_rules[2] and failure_type == "error":
            return message
        if regex is regex_rules[3] and failure_type == "AssertionError":
            return f"{desc.group(1)}: {desc.group(2)}"
        if regex is regex_rules[4] and failure_type == "RuntimeError":
            return desc.group(1)
        if regex is regex_rules[5] and failure_type == "GradcheckError":
            return desc.group(1)
        if regex is regex_rules[6]:
            return f"{desc.group(1)} due to secondary failure {desc.group(3)}"
        if regex is regex_rules[7] and failure_type == "RuntimeError":
            return desc.group(1)
        if regex is regex_rules[8] and failure_type == "AssertionError":
            return desc.group(1)
    return message


def parse_test_suite(test_module, suite, log_list, writer):
    test_suite_summary = Summary()
    test_suite_summary.module_name = suite.name
    test_suite_summary.duration = suite.time
    # TODO: return a dict/list of failure_desc, frequency and module name from here
    # handle case
    for case in suite:
        test_name = case.name
        test_class = case.classname

        verdict = "passed"
        failure_type = ""
        failure_description = ""
        stack_trace = ""
        error = ""
        if case.result:
            if isinstance(case.result[0], Failure):
                test_suite_summary.num_fail += 1
                verdict = "failed"
                message = case.result[0].message.replace("&#x27", "").split("\n")
                try:
                    error = [i for i in error_strings if i in case.result[0].message][-1]
                except:
                    error = "Unknown Error"
                failure_type = error
                if args.vv:
                    stack_trace = (
                        case.result[0]
                        ._elem.text.replace("&lt;", "")
                        .replace("&gt;", "")
                        .replace("&quot;", "")
                        .replace("&#x27", "")
                    )
                    stack_trace = re.sub("    +", "\n", stack_trace)[:255]
                if args.v or args.vv:
                    failure_description = get_fail_desc(failure_type, case.result[0].message)
            elif isinstance(case.result[0], Skipped):
                verdict = "skipped"
                # Check if it is instead an xfail
                if case.result[0].type == "pytest.xfail":
                    test_suite_summary.num_xfail += 1
                    verdict = "xfail"
                    failure_type = "xfail"
                else:
                    test_suite_summary.num_skip += 1
                if args.v or args.vv:
                    failure_description = case.result[0].message
            elif isinstance(case.result[0], Error):
                verdict = "error"
                test_suite_summary.num_error += 1
                for log in log_list:
                    if log.name == test_name:
                        failure_type = ""
                        try:
                            regex = r"(.*\.py:\d+):(.*CRASHED with signal (\d+))"
                            desc = re.search(regex, log.description)
                            failure_type = desc.group(2)
                        except:
                            regex = r":(.*):(.*CRASHED with signal (\d+))"
                            desc = re.search(regex, log.description)
                            failure_type = desc.group(2)
                            # log.description

                        # failure_type = log.stack
                        if args.v or args.vv:
                            if "signal 11" in failure_type:
                                failure_description = "Segmentation Fault"
                            else:
                                message = ""
                                try:
                                    message = log.stderr.splitlines()[-1]
                                    if "terminate called" in message:
                                        message = log.stderr.splitlines()[-2]
                                    if "HABANA_LOG" in message:
                                        raise Exception
                                    failure_description = get_fail_desc(verdict, message)
                                except:
                                    failure_description = "Unknown. No error description found in logs"
                        if args.vv:
                            stack_trace = log.stack
        if verdict == "passed":
            test_suite_summary.num_pass += 1
        writer.writerow(
            [
                test_module.name,
                test_class,
                test_name,
                verdict,
                failure_type,
                (failure_description if (args.v or args.vv) else ""),
                (stack_trace if args.vv else ""),
            ]
        )
    return test_suite_summary


def read_console_logs(stdout_log):
    if stdout_log is None:
        return []
    if os.path.getsize(stdout_log) == 0:
        return []

    test_list = []
    with open(stdout_log) as logfile:
        for line in logfile:
            # Loop till test results
            if "FAILURES" in line:
                break

        # Get results
        line_seek = False
        while line != "":
            # This is a workaround for signal 11, where test may exit at any point
            # making regex matches impossible
            if not line_seek:
                line = logfile.readline()
            line_seek = False
            # Begin of test log in console
            test_error = False
            test_break = False
            regex = r"(\_*\s+)(Test(.*)HPU:0)\.(test_(.*)hpu:0(\S*))(\s+\_*)"
            if re.match(regex, line):
                test_error = True
                match = re.search(regex, line)
                test_class = match.group(2)
                test_name = match.group(4)
                stdout = ""
                stderr = ""
                stack = ""
                description = ""

                # Log only if test has crashed. OTher failures are already captured by the XML
                while "CRASHED with signal" not in line and test_error and not test_break:
                    line = logfile.readline()
                    if not line:
                        test_break = True
                        break
                    if re.match(regex, line):
                        test_error = False
                        line_seek = True
                        break
                    if "self" in line:
                        test_error = False
                        break
                if test_error:
                    while "captured stdout" not in line and not test_break:
                        if test_break:
                            break
                        if line != "\n":
                            description = description + line
                        line = logfile.readline()
                        if not line:
                            test_break = True
                            break
                        if re.match(regex, line):
                            test_break = True
                            line_seek = True
                            break
                    while "captured stderr" not in line and not test_break:
                        if "captured stdout" in line:
                            line = logfile.readline()
                            if not line:
                                test_break = True
                                break
                            if re.match(regex, line):
                                test_break = True
                                line_seek = True
                                break
                        if line != "\n":
                            stdout = stdout + line
                        line = logfile.readline()
                        if not line:
                            test_break = True
                            break
                        if re.match(regex, line):
                            test_break = True
                            line_seek = True
                            break
                    while "frame" not in line and not test_break:
                        if "captured stderr" in line:
                            line = logfile.readline()
                            if not line:
                                test_break = True
                                break
                            if re.match(regex, line):
                                test_break = True
                                line_seek = True
                                break
                        if line != "\n":
                            stderr = stderr + line
                        line = logfile.readline()
                        # EOF
                        if not line:
                            test_break = True
                            # Handles segfault tests where new test case has begun and
                            # no regex pattern can be matched to terminate previous test log
                        if re.match(regex, line):
                            test_break = True
                            line_seek = True
                            break
                    while "frame" in line and not test_break:
                        if line != "\n":
                            stack = stack + line
                        line = logfile.readline()
                        # EOF
                        if not line:
                            test_break = True
                            # Handles segfault tests where new test case has begun and
                            # no regex pattern can be matched to terminate previous test log
                        if re.match(regex, line):
                            test_break = True
                            line_seek = True
                            break
                test_list.append(TestResult(test_name, test_class, description, stack, stdout, stderr))
    return test_list


def get_filenames(args):
    # input xml list
    xml_list = []
    if args.xmlpath.endswith(".xml") and os.path.isfile(args.xmlpath):
        xml_list = [args.xmlpath]
    elif os.path.isdir(args.xmlpath):
        for file in os.listdir(args.xmlpath):
            if file.endswith(".xml"):
                xml_list.append(os.path.join(args.xmlpath, file))
    else:
        print_format(bcolors.FAIL, f"ERROR: Invalid xml path: {args.xmlpath}")
    xml_list.sort()

    # input console logs
    console_list = []
    if args.console_path:
        if args.console_path.endswith(".log") and os.path.isfile(args.console_path):
            console_list = [args.console_path]
        elif os.path.isdir(args.console_path):
            for root, _dir, files in os.walk(args.console_path):
                for log in files:
                    console_list.append(os.path.join(root, log))

    # associate xmls and console logs according to module name
    test_modules = []
    for xml in xml_list:
        test_module = os.path.splitext(os.path.basename(xml))[0]
        try:
            stdout = [i for i in console_list if test_module + "_stdout.log" in i][0]
        except:
            # This module did not dump a console log.
            stdout = None
        print(f"{test_module}:\t{xml},\t{stdout}")
        test_modules.append(TestModule(test_module, xml, stdout))

    # output csv list
    csv_list = []
    if not os.path.isdir(args.outpath) and not args.outpath.endswith(".csv"):
        os.makedirs(args.outpath)
    if args.single_csv:
        csv_list = [
            (
                (
                    os.path.join(args.outpath, args.csvfile)
                    if args.csvfile.endswith(".csv")
                    else (os.path.join(args.outpath, args.csvfile + ".csv"))
                )
                if args.csvfile
                else (os.path.join(args.outpath, "generated_csvfile.csv"))
            )
        ]
    else:
        for module in test_modules:
            csv_list.append(os.path.join(args.outpath, module.name + ".csv"))
    # Test summary
    test_summary = []
    if args.single_csv:
        csv_file = csv_list[0]
        with open(csv_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "test_module",
                    "test_class",
                    "test_name",
                    "status",
                    "failure type",
                    "failure description" if (args.v or args.vv) else "",
                    "stack_trace" if (args.vv) else "",
                ]
            )
            for test_module in test_modules:
                print(f"Processing: {test_module.name}")
                # Get data
                test_summary.extend(generate_report(test_module, writer))
    else:
        for csv_file in csv_list:
            with open(csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "test_module",
                        "test_class",
                        "test_name",
                        "status",
                        "failure type",
                        "failure description" if (args.v or args.vv) else "",
                        "stack_trace" if (args.vv) else "",
                    ]
                )
                for test_module in test_modules:
                    print(f"Processing: {test_module.name}")
                    test_summary.extend(generate_report(test_module, writer))

    get_summary(test_modules, test_summary, os.path.join(args.outpath, args.summary_file), args.summary_level)
    if args.generate_from_console_logs:
        create_fallback_analytics_from_consolelogs(args.outpath, args.console_path)


"""
create_fallback_analytics_from_consolelogs()
create the fallback analytics from console logs
- creates a table and csv file with the count of aten ops that are not supported on HPU
"""


def create_fallback_analytics_from_consolelogs(output_path, console_path):
    from collections import Counter

    matching_lines = []
    pattern = re.compile(r"RuntimeError:(.*?)is not yet supported on HPU")
    for filename in os.listdir(console_path):
        if filename.endswith(".log"):  # Consider only .log files s
            filepath = os.path.join(console_path, filename)
            try:
                with open(filepath, encoding="utf-8") as f:  # Handle potential encoding issues
                    for line in f:
                        if pattern.search(line):
                            stripped_line = pattern.search(line).group(1).strip()
                            # print(f"File : {f} , Line : {stripped_line}")
                            matching_lines.append(stripped_line)  # Remove leading/trailing whitespace

                # Count occurrences of each unique line
                line_counts = Counter(matching_lines)
                sorted_line_counts = sorted(line_counts.items(), key=lambda item: item[1], reverse=True)
                total_count = sum(line_counts.values())
                print(f"Total failback aten ops count: {total_count}")
                # Write the results to the CSV file
                ops_failure_data_file = os.path.join(output_path, "aten_ops_failure_data_console.csv")
                with open(ops_failure_data_file, "w", newline="", encoding="utf-8") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(["Line", "Count"])
                    for line, count in sorted_line_counts:
                        csvwriter.writerow([line, count])
                    csvwriter.writerow(["Total", total_count])
            except Exception as e:
                print(f"Error processing file {filename}: {e}, {e.args}")


"""
create_fallback_analytics_from_consolelogs()
create the fallback analytics from generated csv file
- creates a table and csv file with the count of aten ops that are not supported on HPU
TODO : we need to modify our generated_csv file logic as it fails to capture finer details
 in addtion add more additional column for dtype information capture.
"""


def create_ops_fallback_analytics_file(output_path, generated_csv_file):
    import pandas as pd

    df = pd.read_csv(generated_csv_file)
    prefix = "RuntimeError: aten::"
    suffix = "is not yet supported"
    fallback_errors = df[
        (df["status"] == "failed") & (df["failure description"].str.contains("is not yet supported on HPU", na=False))
    ]
    if len(fallback_errors) == 0:
        print("No fallback failures reported, Report aten_ops_failure_data.csv not created")
        return

    op_list_filtered = fallback_errors["failure description"].str.split(suffix).str[0].str.split(prefix).str[1]
    unique_ops = op_list_filtered.unique()
    ops_data = pd.DataFrame()
    for _index, op in enumerate(unique_ops):
        count = op_list_filtered[op_list_filtered == op].count()
        new_row = pd.DataFrame({"op": [op], "count": count})
        ops_data = pd.concat([ops_data, new_row], ignore_index=True)
    ops_failure_data_file = os.path.join(output_path, "aten_ops_failure_data.csv")
    print(f"Creating ops failure report in : {ops_failure_data_file}, Total Items : {len(ops_data)}")
    sorted_ops_data = ops_data.sort_values(by="count", ascending=False)
    column_sum = sorted_ops_data["count"].sum()
    new_row = pd.DataFrame({"op": ["Total"], "count": [column_sum]})
    sorted_ops_data = pd.concat([sorted_ops_data, new_row], ignore_index=True).dropna()
    sorted_ops_data.to_csv(ops_failure_data_file, index=False, encoding="utf-8")


summary_rules = [
    r"(FAILED\s+(.*))",  # Failed tests
    r"(SKIPPED\s+\[(\d+).(\d+)s\]\s+\[(\d+)\]\s+(.*\.py):(\d+:)?\s+(.*))",
    r"(XFAIL\s+(.*))",  # XFAIL tests primary
]


def get_summary_lists(logfile, summary_level):
    skip_reasons = {}
    xfail_reasons = {}
    # Template:
    # Start at =========================== short test summary info ============================
    # SKIPPED [1] test_autograd.py:7919: Only runs on cuda
    # XFAIL test_autograd.py::TestAutogradDeviceTypeHPU::test_GRU_grad_and_gradgrad_hpu
    #   reason: [NOTRUN] https://jira.habana-labs.com/browse/SW-58497
    # FAILED test_autograd.py::TestAutogradDeviceTypeHPU::test_inplace_on_view_python_hpu
    # End at ==== 2 failed, 25 passed, 16 skipped, 476 deselected, 25 xfailed in 11.12s =====
    summary_block = False
    for line in logfile:
        while summary_block is False and line != "":
            if "short test summary info" in line:
                summary_block = True
                break
            line = logfile.readline()
        for regex in summary_rules:
            desc = re.search(regex, line)
            if desc is None:
                continue
            elif regex is summary_rules[1] and summary_level & SummaryLevel.SKIP:
                # Skipped tests
                add_key_to_dict(desc.group(7), skip_reasons, desc.group(4))
            elif regex is summary_rules[2] and summary_level & SummaryLevel.XFAIL:
                # xfailed tests
                line = logfile.readline()
                # Xfail tests secondary
                regex = r"(^\s+reason:\s+)(\[NOTRUN\]\s+(.*))"
                desc = re.search(regex, line)
                if desc is not None:
                    add_key_to_dict(desc.group(3), xfail_reasons, 1)
                else:
                    add_key_to_dict("REASON MISSING", xfail_reasons, 1)
            else:
                # TODO: do something here . check what is going on in logs
                break
    return xfail_reasons, skip_reasons


def add_key_to_dict(key, dictionary, frequency):
    if key in dictionary:
        dictionary[key] += int(frequency)
    else:
        dictionary[key] = int(frequency)


def update_fail_dict_list(dictionary, dict, module):
    for element in dict.keys():
        frequency = dict[element]
        module_list = [module]
        if element in dictionary.keys():
            frequency += dictionary[element].frequency
            module_list = list(set([module] + dictionary[element].module))
            dictionary.update({element: DictListValues(module_list, frequency)})
        else:
            dictionary.update({element: DictListValues([module], frequency)})
    return 0


def print_big(fail_dict_list):
    for element in fail_dict_list:
        print(f"{element}: {fail_dict_list[element].frequency}, {set(fail_dict_list[element].module)}")


def get_summary(test_modules, test_summary, summary_file, summary_level=0xF):
    # Complete test summary
    # summary_to_file(test_summary, summary_file)
    # Specific fails
    skip_dict_list = {}
    xfail_dict_list = {}
    for module in test_modules:
        # Read all logfiles. Check if logfile present
        if module.stdout is None:
            print_format(bcolors.WARNING, f"WARNING: logfile not found for {module.stdout}")
            continue
        # maintain a dict to get frequency of fails and their module names
        with open(module.stdout) as logfile:
            # reason, count, module
            print(f"Processing {module.name} summary")
            xfail_dict, skip_dict = get_summary_lists(logfile, summary_level)
            import pprint

            pprint.pprint(skip_dict)
            # Collate dicts to master dict with module names
            # and dump info to respective files
            # TODO: fail file
            if summary_level & SummaryLevel.XFAIL:
                # xfail file
                update_fail_dict_list(xfail_dict_list, xfail_dict, module.name)
                dump_dict_to_csv(
                    xfail_dict_list, f'{os.path.join(args.outpath, args.summary_file.split(".csv")[0])}_xfail.csv'
                )
            if summary_level & SummaryLevel.SKIP:
                # skip files
                update_fail_dict_list(skip_dict_list, skip_dict, module.name)
                dump_dict_to_csv(
                    skip_dict_list, f'{os.path.join(args.outpath, args.summary_file.split(".csv")[0])}_skip.csv'
                )
    return 0


def dump_dict_to_csv(dict, file):
    # Sorting the dict wrt frequency of reasons
    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    with open(file, "w") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["reason", "frequency", "module"])
        for reason in dict:
            writer.writerow([reason, dict[reason].frequency, dict[reason].module])
    return 0


def summary_to_file(test_summary, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Pytorch Python Tests",
                "Total Tests",
                "Pass Tests",
                "Error Tests",
                "xFailed Tests",
                "Failed Tests",
                "Skipped Tests",
                "Duration",
            ]
        )
        for item in test_summary:
            # Write only relevant modules
            if item.total() != 0:
                writer.writerow(item.to_list())
    return 0


def generate_report(test_module, writer):
    # handle suite
    test_suite_summary = []
    if args.vv:
        log_list = read_console_logs(test_module.stdout)
    else:
        log_list = []
    xml = JUnitXml.fromfile(test_module.xml)
    if xml._tag == "testsuites":
        for suite in xml:
            test_suite_summary.append(parse_test_suite(test_module, suite, log_list, writer))
    elif xml._tag == "testsuite":
        test_suite_summary.append(parse_test_suite(test_module, xml, log_list, writer))
    else:
        print_format(bcolors.FAIL, "ERROR. Invalid xml format")
        return -1
    return test_suite_summary


def get_args():
    parser = argparse.ArgumentParser(description="Parse JUnit xml and generate concise csv")
    parser.add_argument("-x", "--xmlpath", required=False, help="path to xml file(s) generated by pytest.")
    parser.add_argument(
        "-s",
        "--single_csv",
        default=True,
        action="store_false",
        help="Collect different xml results in single csv. Default: True",
    )
    parser.add_argument(
        "--generate_from_console_logs",
        default=True,
        action="store_false",
        help="Generate fallback analytics from console logs",
    )
    parser.add_argument("-c", "--csvfile", default="generated_csvfile.csv", help="csvfile path")
    parser.add_argument("-o", "--outpath", required=False, default=".", help="outfile path")
    parser.add_argument("-con", "--console_path", required=False, help="path to console logs")
    parser.add_argument(
        "-sf",
        "--summary_file",
        default="pytorch_test_report.csv",
        help='Filename of summary file. Default = "pytorch_test_report.csv"',
    )
    parser.add_argument(
        "-sl",
        "--summary_level",
        default=0xF,
        type=int,
        help="Types of outcomes to include in summary. Default: summary + fails + skip + xfails",
    )
    parser.add_argument(
        "-v",
        "-v",
        default=False,
        action="store_true",
        help="Includes more details in csv" "Adds failure details to the csv",
    )
    parser.add_argument(
        "-vv",
        "-vv",
        default=False,
        action="store_true",
        help="Includes even more details in csv" "Adds Stack trace to the csv",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = get_args()
        get_filenames(args)
    except Exception as e:
        print_format(bcolors.FAIL, traceback.format_exc())
