#!/usr/bin/env bash
set -euo pipefail

QWEN_CODE_REPO="https://github.com/QwenLM/Qwen3-VL-Embedding.git"
QWEN_CODE_REVISION="8ce3aab1fbcc7b7143b094f0b2005dd89e7246b9"
QWEN_LOCK_SHA256="1f7b51793717d81984a081eafe097ef41f5a3f7912953a0ea8e158232af54609"
QWEN_UV_VERSION="0.9.26"
QWEN_MODEL_REPO="Qwen/Qwen3-VL-Embedding-2B"
QWEN_MODEL_REVISION="2a50926d213628c727f38025982a76f655673f54"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
QWEN_DIR="${QWEN3VL_REPO:-${PROJECT_DIR}/.external/Qwen3-VL-Embedding}"
QWEN_MODEL_DIR="${QWEN3VL_MODEL:-${QWEN_DIR}/models/Qwen3-VL-Embedding-2B}"

for command_name in awk git sha256sum uv; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
done

actual_uv_version="$(uv --version | awk '{print $2}')"
if [[ "${actual_uv_version}" != "${QWEN_UV_VERSION}" ]]; then
    echo "Expected uv ${QWEN_UV_VERSION}, found ${actual_uv_version}" >&2
    echo "Install the tested version with: python -m pip install uv==${QWEN_UV_VERSION}" >&2
    exit 1
fi

if [[ -e "${QWEN_DIR}" && ! -d "${QWEN_DIR}/.git" ]]; then
    echo "Refusing to overwrite non-Git path: ${QWEN_DIR}" >&2
    exit 1
fi

if [[ ! -d "${QWEN_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${QWEN_DIR}")"
    git clone "${QWEN_CODE_REPO}" "${QWEN_DIR}"
fi

if [[ -n "$(git -C "${QWEN_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to change a Qwen3-VL checkout with tracked local changes: ${QWEN_DIR}" >&2
    exit 1
fi

git -C "${QWEN_DIR}" fetch origin "${QWEN_CODE_REVISION}"
git -C "${QWEN_DIR}" checkout --detach "${QWEN_CODE_REVISION}"

actual_lock_sha256="$(sha256sum "${QWEN_DIR}/uv.lock" | awk '{print $1}')"
if [[ "${actual_lock_sha256}" != "${QWEN_LOCK_SHA256}" ]]; then
    echo "Unexpected Qwen3-VL uv.lock checksum: ${actual_lock_sha256}" >&2
    exit 1
fi

QWEN_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${QWEN_REQUIREMENTS}"' EXIT

# Export the upstream lock, then install the exact locked artifacts from HTTPS
# PyPI. The pinned upstream lock records an insecure HTTP mirror, so uv sync is
# intentionally not used here.
uv export \
    --project "${QWEN_DIR}" \
    --frozen \
    --no-emit-project \
    --output-file "${QWEN_REQUIREMENTS}" \
    >/dev/null

uv venv --python 3.11 --clear "${QWEN_DIR}/.venv"
env \
    -u UV_INDEX \
    -u UV_DEFAULT_INDEX \
    -u UV_INDEX_URL \
    -u UV_EXTRA_INDEX_URL \
    -u PIP_INDEX_URL \
    -u PIP_EXTRA_INDEX_URL \
    uv pip install \
        --python "${QWEN_DIR}/.venv/bin/python" \
        --require-hashes \
        --no-config \
        --default-index https://pypi.org/simple \
        --requirements "${QWEN_REQUIREMENTS}"

if [[ -n "${QWEN3VL_MODEL:-}" ]]; then
    if [[ ! -d "${QWEN_MODEL_DIR}" ]]; then
        echo "QWEN3VL_MODEL is not a directory: ${QWEN_MODEL_DIR}" >&2
        exit 1
    fi
else
    "${QWEN_DIR}/.venv/bin/hf" download \
        "${QWEN_MODEL_REPO}" \
        --revision "${QWEN_MODEL_REVISION}" \
        --local-dir "${QWEN_MODEL_DIR}"
fi

echo
echo "Qwen3-VL is ready at: ${QWEN_DIR}"
echo "Export this before running Meric:"
echo "export QWEN3VL_REPO=\"${QWEN_DIR}\""
if [[ -n "${QWEN3VL_MODEL:-}" ]]; then
    echo "export QWEN3VL_MODEL=\"${QWEN_MODEL_DIR}\""
fi
