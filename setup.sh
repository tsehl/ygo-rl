#!/usr/bin/env bash
# Installe les dépendances externes (non versionnées) et compile l'oracle ygopro-core.
# Linux ou macOS. Idempotent : relançable sans risque.
set -euo pipefail
cd "$(dirname "$0")"

# dépôts externes, épinglés aux commits validés contre le moteur JAX
clone() {  # url dossier commit [récursif]
  if [ ! -d "$2/.git" ]; then
    git init -q "$2"
    git -C "$2" remote add origin "$1"
  fi
  if [ "$(git -C "$2" rev-parse HEAD 2>/dev/null)" != "$3" ]; then
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q FETCH_HEAD
  fi
  if [ "${4:-}" = rec ]; then git -C "$2" submodule update -q --init --recursive --depth 1; fi
}
clone https://github.com/edo9300/ygopro-core oracle/ygopro-core 6864373c24d7313c8e451df17877c12c4f713c46 rec
clone https://github.com/ProjectIgnis/CardScripts.git data/CardScripts bc5d82d66842f3c62d20747c530b164ce98004f3
clone https://github.com/ProjectIgnis/BabelCDB.git data/BabelCDB 2e6f985a5537d9b16ec7396787b5279d94af419d

# premake5
if ! command -v premake5 >/dev/null; then
  if [ "$(uname)" = Darwin ]; then
    brew install premake
  else
    mkdir -p .tools
    curl -sL https://github.com/premake/premake-core/releases/download/v5.0.0-beta2/premake-5.0.0-beta2-linux.tar.gz \
      | tar xz -C .tools
    export PATH="$PWD/.tools:$PATH"
  fi
fi

# libocgcore partagée
lib=oracle/ygopro-core/bin/release/libocgcore.$([ "$(uname)" = Darwin ] && echo dylib || echo so)
if [ ! -f "$lib" ]; then
  (cd oracle/ygopro-core && premake5 gmake2 && make -C build ocgcoreshared config=release -j"$(getconf _NPROCESSORS_ONLN)")
fi
python3 oracle/smoke.py

# uv (lancement des scripts : uv run --with jax python engine/compare.py ...)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
