import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

PT_WHITELIST = [
    "libtorch.so",
    "libc10.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libaeon.so.1",
    "libhabana_pytorch_plugin.so",  # 2 copies of library causes bridge issue
]

for p in POLICIES:
    p["lib_whitelist"].extend(PT_WHITELIST)

if __name__ == "__main__":
    sys.exit(main())
