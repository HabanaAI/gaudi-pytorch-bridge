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
import logging
import subprocess as sp  # nosec
import sys
import tempfile

log = logging.getLogger(__file__)


def ensure_icecc_setup():
    lsb_release = sp.check_output(["lsb_release", "-d"], text=True)
    if "Ubuntu" not in lsb_release and "Debian" not in lsb_release:
        log.fatal("--use-icecc flag only supported for dpkg-based distros")
        sys.exit(1)

    icecc_installed = call_with_error_logging("dpkg -s icecc") == 0

    if icecc_installed:
        ensure_iceccd_started()
    else:
        log.info("icecc not installed. Installing and doing setup...")
        sp.check_call(["sudo", "apt", "update"])
        sp.check_call(["sudo", "apt", "install", "icecc", "-y"])
        sp.check_call(
            [
                "sudo",
                "sed",
                "-i",
                's/ICECC_NICE_LEVEL="5"/ICECC_NICE_LEVEL="10"/',
                "/etc/icecc/icecc.conf",
            ]
        )
        sp.check_call(["sudo", "systemctl", "restart", "iceccd"])


def call_with_error_logging(cmd):
    with tempfile.TemporaryFile() as tmp_out, tempfile.TemporaryFile() as tmp_err:
        try:
            return sp.call(cmd.split(), stdout=tmp_out, stderr=tmp_err)
        except Exception:
            log.error(f"Unexpected error when calling `{cmd}`:")
            log.error("stdout:")
            log.error(tmp_out.readlines())
            log.error("stderr:")
            log.error(tmp_err.readlines())
            raise


def ensure_iceccd_started():
    iceccd_stopped = call_with_error_logging("systemctl status iceccd") != 0

    if iceccd_stopped:
        log.info("iceccd was stopped. Trying to start it...")
        sp.check_call(["sudo", "systemctl", "start", "iceccd"])
