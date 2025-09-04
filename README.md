# Intel® Gaudi® PyTorch Bridge

Intel Gaudi PyTorch Bridge consists of several Python packages enabling Intel Gaudi
functionality in PyTorch with minimal code changes.

## Repository Build

This repository can be built as part of the Intel Gaudi software stack or as a standalone project. The instructions in this README focus on the standalone installation using the latest Intel Gaudi software release.

These steps assume you are building on Ubuntu 22.04. If you use a different OS, you need to adjust the package installation steps based on the instructions provided [here](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#driver-installation).

### One-time Setup

Follow the below steps once to configure your environment for the repository build.

1. Install the Intel Gaudi software and driver using steps from the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#driver-installation). For example:

```bash
sudo apt update && sudo apt install -y curl gnupg pciutils wget
wget 'https://vault.habana.ai/artifactory/gaudi-installer/latest/habanalabs-installer.sh'
bash habanalabs-installer.sh install -t base -y
```

2. Prepare the Intel Gaudi PyTorch bridge repository and install a proper version of the Gaudi-enabled `torch` wheel:

```bash
export HABANA_SOFTWARE_STACK="$(pwd)"
export PYTORCH_MODULES_ROOT_PATH="$HABANA_SOFTWARE_STACK/gaudi-pytorch-bridge"
git clone git@github.com:HabanaAI/gaudi-pytorch-bridge.git $PYTORCH_MODULES_ROOT_PATH

IFS=- read -r VERSION BUILD <<EOF
$(bash habanalabs-installer.sh -v)
EOF
"${PYTORCH_MODULES_ROOT_PATH}"/scripts/install_torch_fork.sh "$VERSION" "$BUILD"
```

3. Install the requirements:
```bash
pip install -r "$PYTORCH_MODULES_ROOT_PATH"/requirements.txt
pip install habana-media-loader==$VERSION.$BUILD
```

4. Allow the build command to install artifacts:

```bash
sudo chmod +xw /usr/lib/habanalabs
sudo ln -s /usr/include/habanalabs/hl_logger /usr/include/habanalabs/hl_logger/include

```

### Code Build

Once the one-time setup is complete, you can configure the necessary environment variables and run the build by following the below steps:

1. Set up source and binary directories used for building the Intel Gaudi PyTorch bridge:
```bash
export HABANA_SOFTWARE_STACK="$(pwd)"

export HCL_INCLUDE_DIR=/usr/include/habanalabs/
export MEDIA_ROOT=$(python -c "import habana_frameworks.mediapipe, os;print(os.path.dirname(habana_frameworks.mediapipe.__file__))")
export SPECS_EXT_ROOT=/usr/include/habanalabs/
export SYNAPSE_INCLUDE_DIR=/usr/include/habanalabs/
export SYNAPSE_UTILS_INCLUDE_DIR=/usr/include/habanalabs/
export SWTOOLS_SDK_ROOT=/usr/include/habanalabs/

export BUILD_ROOT="$HOME/builds"
export BUILD_ROOT_LATEST=/usr/lib/habanalabs/
export PYTORCH_MODULES_RELEASE_BUILD="$BUILD_ROOT/pytorch_modules_release"  # the release build artifact directory
export PYTORCH_MODULES_DEBUG_BUILD="$BUILD_ROOT/pytorch_modules_debug"  # the debug build artifact directory
export PYTORCH_MODULES_ROOT_PATH="$HABANA_SOFTWARE_STACK/gaudi-pytorch-bridge"
```

2. Build the Intel Gaudi PyTorch bridge:
```bash
"$PYTORCH_MODULES_ROOT_PATH"/.devops/build.py -cir
```
**Notes:**
- The `-i` flag installs the wheels after they are built.
- It is recommended to leverage CCache and Icecream for faster compilation. Icecream (icecc) allows using a much larger parallel job count (`-j N`). The `N` depends on your compute cluster size.
- Sometimes the final build command is interrupted while preparing the environment. In this case you can add `--recreate-venv force` to resolve any potential issues.

### Running tests
After building the code, you can run tests to validate functionality.

1. Load the commands to run the tests:
```bash
source $PYTORCH_MODULES_ROOT_PATH/.ci/scripts/build.sh
```

2. Install the test requirements:
```bash
pip install -r $PYTORCH_MODULES_ROOT_PATH/.ci/requirements/requirements-test.txt
```
These are required in case you want to run Python-based tests in addition to C++ tests.

3. Run the tests:
```bash
run_pytorch_modules_tests
```
To run tests on a specific device, use the --dut flag (e.g., --dut gaudi3).
You can also specify subsets using --pytest-mode - to select the desired test mode, and --suite-type - to choose the specific test suite to run.

A device is required to execute the tests.
