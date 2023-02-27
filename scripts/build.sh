#!/usr/bin/env bash
# set -x             # for debug
set -euo pipefail  # fail early
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DOC_BUILD_DIR="${SCRIPT_DIR}/../doc-build"
DOCUSAURUS_DIR="${SCRIPT_DIR}/../docusaurus"

# build docs
cd "${DOC_BUILD_DIR}"
docer extract -o hpcaitech -p ColossalAI
docer docusaurus -d ../docusaurus
docer autodoc -o hpcaitech -p ColossalAI

# build html
cd "${DOCUSAURUS_DIR}"
yarn install
yarn build