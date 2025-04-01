# Intel® Gaudi® PyTorch Bridge

Intel Gaudi PyTorch Bridge consists of several Python packages enabling Intel Gaudi
functionality in PyTorch with minimal code changes.

## Repository Build

This repository can be built as part of the Intel Gaudi software stack or as a standalone project. The instructions in this README focus on the standalone installation using the latest Intel Gaudi software release.

These steps assume you are building on Ubuntu 22.04. If you use a different OS, you need to adjust the package installation steps based on the instructions provided [here](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#driver-installation).

### One-time Setup

Follow the below steps once to configure your environment for the repository build.

1. If you are not using a prebuilt Intel Gaudi container, install the Intel Gaudi software and driver by following the steps from the [Installation Guide]
(https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#driver-installation). For example:

```bash
sudo apt update && sudo apt install -y curl gnupg pciutils wget
wget 'https://vault.habana.ai/artifactory/gaudi-installer/latest/habanalabs-installer.sh'
bash habanalabs-installer.sh install -t base -y

sudo ln -s /usr/lib/habanalabs/libaeon.so.1 /usr/lib/habanalabs/libaeon.so
```

If you are using a custom container provided in the Intel Gaudi vault, download the habanalabs-installer.sh without running it:

```bash
sudo apt update && sudo apt install -y curl gnupg pciutils wget
wget 'https://vault.habana.ai/artifactory/gaudi-installer/latest/habanalabs-installer.sh'
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

3. Set up the required 3rd party code:
```bash
mkdir 3rd-parties
pushd 3rd-parties

git clone --depth 1 https://github.com/abseil/abseil-cpp.git
git clone --depth 1 --branch 9.1.0 https://github.com/fmtlib/fmt fmt-9.1.0
git clone --depth 1 --branch 3.3.9 https://gitlab.com/libeigen/eigen.git
git clone --depth 1 --branch v0.8.1 https://github.com/Neargye/magic_enum.git magic_enum-0.8.1
git clone --depth 1 --branch v1.13.0 https://github.com/google/googletest.git googletest_1_13

git clone --depth 1 --branch v3.4.0 https://github.com/nlohmann/json.git
sed -i 's/namespace nlohmann/namespace nlohmannV340/; s/nlohmann::/nlohmannV340::/g' json/single_include/nlohmann/json.hpp

popd
```

4. Set up additional build dependencies:
```bash
git clone --depth 1 --branch 1.20.0-543 https://github.com/HabanaAI/HCL.git
git clone --depth 1 --branch main https://github.com/HabanaAI/Intel_Gaudi3_Software.git

sudo ln -s /usr/include/habanalabs/ /usr/include/habanalabs/include
```

5. Patch the `Intel_Gaudi3_Software` repository:
```bash
patch -p1 <$PYTORCH_MODULES_ROOT_PATH/.devops/patches/Intel_Gaudi3_Software.patch
```

6. Install the media interface:
```bash
sudo cp $PYTORCH_MODULES_ROOT_PATH/.devops/patches/media_pytorch_proxy.h /usr/include/habanalabs/media_pytorch_proxy.h
```

### Code Build

Once the one-time setup is complete, you can configure the necessary environment variables and run the build by following the below steps:

1. Set up source and binary directories used for building the Intel Gaudi PyTorch bridge:
```bash
export HABANA_SOFTWARE_STACK="$(pwd)"
export THIRD_PARTIES_ROOT="$HABANA_SOFTWARE_STACK/3rd-parties"

export HCL_ROOT="$HABANA_SOFTWARE_STACK/HCL/hcl/"
export HL_LOGGER_INCLUDE_DIRS="$HABANA_SOFTWARE_STACK/HCL/dependencies/swtools_sdk/hl_logger/include;$THIRD_PARTIES_ROOT"
export MEDIA_ROOT=/usr/include/habanalabs/
export SPECS_EXT_ROOT="$HABANA_SOFTWARE_STACK/Intel_Gaudi3_Software/specs_external/"
export SYNAPSE_ROOT=/usr/include/habanalabs/
export SYNAPSE_UTILS_ROOT=/usr/include/habanalabs/

export BUILD_ROOT="$HOME/builds"
export BUILD_ROOT_LATEST=/usr/lib/habanalabs/
export PYTORCH_MODULES_RELEASE_BUILD="$BUILD_ROOT/pytorch_modules_release"  # the release build artifact directory
export PYTORCH_MODULES_DEBUG_BUILD="$BUILD_ROOT/pytorch_modules_debug"  # the debug build artifact directory
export PYTORCH_MODULES_ROOT_PATH="$HABANA_SOFTWARE_STACK/gaudi-pytorch-bridge"
```

2. Install the requirements:
```bash
pip install -r "$PYTORCH_MODULES_ROOT_PATH"/requirements.txt
```

3. Build the Intel Gaudi PyTorch bridge:
```bash
"$PYTORCH_MODULES_ROOT_PATH"/.devops/build.py -cir
```
**Notes:**
- The `-i` flag installs the wheels after they are built.
- It is recommended to leverage CCache and Icecream for faster compilation. Icecream (icecc) allows using a much larger parallel job count (`-j N`). The `N` depends on your compute cluster size.
- Sometimes the final build command is interrupted while preparing the environment. In this case you can add `--recreate-venv force` to resolve any potential issues.

### Usage

Once the code building is complete it can be used as shown below.
```
import torch # Loads package destined for running on CPU
import habana_frameworks.torch # Loads HPU plugin
```
Using package habana_frameworks is the desired way of accessing code.
