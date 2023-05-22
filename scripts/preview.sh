#!/usr/bin/env bash
# set -x             # for debug
set -euo pipefail  # fail early

OWNER=${1:-hpcaitech}
REPO=${2:-ColossalAI}
REF=${3:-main}
CACHE_DIR=${4:-$PWD/.cache}

rm -rf $CACHE_DIR

# get directory
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DOC_BUILD_DIR="${SCRIPT_DIR}/../doc-build"
DOCUSAURUS_DIR="${SCRIPT_DIR}/../docusaurus"

# build docs
cd "${DOC_BUILD_DIR}"
docer extract -o $OWNER -p $REPO -r $REF -c $CACHE_DIR
docer autodoc -o $OWNER -p $REPO -n current -r $REF -c $CACHE_DIR
docer docusaurus -d ../docusaurus -c $CACHE_DIR

# build html
cd "${DOCUSAURUS_DIR}"
yarn install
yarn build
