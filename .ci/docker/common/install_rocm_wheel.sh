#!/usr/bin/env bash
# Script used only in CD pipeline.
#
# Installs ROCm from the multi-arch TheRock wheel index instead of OS packages
# or tarballs. The ROCm SDK is pip-installed (it unpacks under
# <site-packages>/_rocm_sdk_core) and its real install root is discovered via
# the rocm-sdk CLI and exported through /etc/rocm_env.sh, mirroring PR #188429.
# We deliberately do NOT create an /opt/rocm symlink: build_env_setup.py and
# repair_wheel.py discover ROCM_HOME from the environment instead.

set -exou pipefail

# https://repo.amd.com/rocm/whl-multi-arch/ hosts the multi-arch ROCm wheels.
THEROCK_INDEX_URL="${THEROCK_INDEX_URL:-https://repo.amd.com/rocm/whl-multi-arch/}"
# Track the 7.14 line from the multi-arch index (latest 7.14.x), no hard pin.
# THEROCK_VERSION is the ROCm minor line (e.g. "7.14"); the spec below resolves
# to the newest 7.14.x on the index.
THEROCK_VERSION="${THEROCK_VERSION:-7.14}"
ROCM_PIP_SPEC="rocm[libraries,devel,device-all]==${THEROCK_VERSION}.*"

echo "=============================================="
echo "ROCm Multi-Arch Wheel Installation (TheRock)"
echo "=============================================="
echo "Index URL:      ${THEROCK_INDEX_URL}"
echo "ROCm spec:      ${ROCM_PIP_SPEC}"
echo "=============================================="

# device-all pulls in the kernels for every supported gfx target so the built
# wheel is multi-arch; libraries+devel provide the runtime libs and the
# headers/hipcc needed to compile PyTorch against ROCm.
python3 -m pip install \
    --index-url "${THEROCK_INDEX_URL}" \
    "${ROCM_PIP_SPEC}"

# Discover the real install root/bin via the rocm-sdk CLI helper (shipped by the
# rocm-sdk-core wheel). This points at <site-packages>/_rocm_sdk_core.
ROCM_HOME="$(rocm-sdk path --root)"
ROCM_BIN="$(rocm-sdk path --bin)"

echo "ROCM_HOME=${ROCM_HOME}"
echo "ROCM_BIN=${ROCM_BIN}"

# theRock bundles system dependencies (libdrm, liblzma, etc.) under
# lib/rocm_sysdeps; expose their libs alongside the core ROCm libs.
ROCM_SYSDEPS_LIB="${ROCM_HOME}/lib/rocm_sysdeps/lib"

# Write the environment file (sourced by CI scripts / interactive shells).
# Paths point at the real wheel install root, not /opt/rocm.
cat > /etc/rocm_env.sh << ROCM_ENV
export ROCM_PATH="${ROCM_HOME}"
export ROCM_HOME="${ROCM_HOME}"
export PATH="${ROCM_BIN}:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_SYSDEPS_LIB}:\${LD_LIBRARY_PATH:-}"
ROCM_ENV

echo "source /etc/rocm_env.sh" >> /etc/bashrc || true

echo "=============================================="
echo "TheRock ROCm wheel install complete"
echo "ROCM_HOME=${ROCM_HOME}"
if [[ -d "${ROCM_SYSDEPS_LIB}" ]]; then
    echo "rocm_sysdeps libs: ${ROCM_SYSDEPS_LIB}"
fi
echo "=============================================="
