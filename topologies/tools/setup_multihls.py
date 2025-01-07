import os
import socket
from contextlib import closing
import time
import datetime
try:
    import pexpect
except Exception:
    print("pexpect is not installed")


def mhprint(*args, **kwargs):
    "function mhprint"
    logpath = os.getenv('HABANA_LOGS')
    mhsetup_log = os.path.join(logpath, "multihls_setup.txt")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    __prompt_string = f"MultiHLS {timestamp}> "
    print(__prompt_string, *args, **kwargs)
    with open(mhsetup_log, "a+") as logf:
        print(__prompt_string, *args, **kwargs, file=logf)


class MultiHLS(object):
    """
    Class contains multihls related helper functions
    """

    @staticmethod
    def check_tcp_port(host, port=3022):
        "check if tcp port is opne or closed"
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(3)
            if sock.connect_ex((host, port)) == 0:
                mhprint(f"port {port} is open on {host}")
                return True
            else:
                mhprint(f"port {port} is closed on {host}")
                return False

    @staticmethod
    def wait_till_timeout(host, port=3022, timeout=400):
        "wait till port is open,i.e. other container is up"
        start_time = time.time()
        while not __class__.check_tcp_port(host, port):
            total = round(time.time() - start_time)
            if total >= timeout:
                mhprint(f"timed out for {host}:{port}")
            assert total < timeout, ("wait till port available"
                                     f" timed out for {host}:{port}")
            time.sleep(15)
        return True

    @staticmethod
    def setup_passwordless_rsh(password='4Hbqa???', sshport=3022, user='root'):
        """
        setup password less  communication b/w containers
        """
        assert os.path.exists("/.dockerenv"), "must run in docker env"
        hls_list = str(os.getenv("MULTI_HLS_IPS", "")).split(',')
        if len(hls_list) <= 1:
            return
        homedir = f'/{user}'
        keyfile = os.path.join(homedir, '.ssh/id_rsa')
        keygen_cmd = f"bash -c \"ssh-keygen -q -t rsa -b 4096 -N '' -f {keyfile} <<<y 2>&1 >/dev/null\""

        if os.path.exists(keyfile) is False:
            mhprint(keygen_cmd)
            output = pexpect.run(keygen_cmd,
                                 events={
                                    "({keyfile}):": f"{keyfile}\n",
                                    "Overwrite (y/n)?": "y\n",
                                    "(empty for no passphrase):": "\n",
                                    "Enter same passphrase again:": "\n",
                                })
            mhprint(output.decode('utf8'))
        for hls in hls_list[1:]:
            hls_node = hls.split('-')[0]
            __class__.wait_till_timeout(hls_node, port=sshport)
            keycopy_cmd = (f"ssh-copy-id -i {keyfile} "
                           "-o StrictHostKeyChecking=no "
                           f"root@{hls_node} -p {sshport} ")
            test_cmd = f"ssh -p {sshport} -o StrictHostKeyChecking=no root@{hls_node} hostname"

            mhprint(test_cmd)
            output, exitstatus = pexpect.run(test_cmd, timeout=20,
                                             withexitstatus=True)
            if exitstatus != 0:
                mhprint(keycopy_cmd)
                output, exitstatus = pexpect.run(keycopy_cmd, timeout=30,
                                                 withexitstatus=True,
                                                 events={
                                                    " password:": f"{password}\n",
                                                 })
                mhprint(output.decode('utf8'))
                assert exitstatus == 0, "cmd failed"
                output, exitstatus = pexpect.run(test_cmd, timeout=20,
                                                 withexitstatus=True)
                assert exitstatus == 0, "cmd failed"
                mhprint(output.decode('utf8'))


if __name__ == "__main__":
    sshport = os.getenv('DOCKER_SSHD_PORT', 3022)
    multihls = MultiHLS()
    multihls.setup_passwordless_rsh(sshport=sshport)
