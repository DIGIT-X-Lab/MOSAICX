#!/usr/bin/env bash
set -euo pipefail
VENV_DIR="${HOME}/.mosaicx-venv"
FULL="" NONINTERACTIVE=""
for arg in "$@"; do
  case "$arg" in
    --full) FULL="--full" ;; --non-interactive) NONINTERACTIVE="--non-interactive" ;;
  esac
done

# Platform detection
case "$(uname -s)" in Darwin) PLATFORM="macOS";; Linux) PLATFORM="Linux";; *) PLATFORM="$(uname -s)";; esac
echo ">> MOSAICX bootstrap ($PLATFORM)"

# Find Python >= 3.11
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ok=$("$candidate" -c 'import sys;print(sys.version_info>=(3,11))' 2>/dev/null||echo False)
    [ "$ok" = "True" ] && PYTHON="$(command -v "$candidate")" && break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "!! Python >= 3.11 not found." >&2
  [ "$PLATFORM" = "macOS" ] && echo "   Install: brew install python@3.13" >&2 \
    || echo "   Install: sudo apt install python3.13 python3.13-venv" >&2
  exit 1
fi
echo "ok python: $PYTHON"
# Resolve uv or pip
USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1
elif [ -n "$FULL" ]; then
  echo ">> installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 && USE_UV=1
fi

# Create / reuse venv
if [ -d "$VENV_DIR" ]; then echo "ok venv exists: $VENV_DIR"
else
  echo ">> creating venv at $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
  echo "ok venv created"
fi

# Install mosaicx
echo ">> installing mosaicx ..."
if [ "$USE_UV" -eq 1 ]; then
  uv pip install --python "${VENV_DIR}/bin/python" mosaicx
else
  "${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null 2>&1
  "${VENV_DIR}/bin/pip" install mosaicx
fi
echo "ok mosaicx installed"
# Add venv/bin to PATH in shell rc
BIN_DIR="${VENV_DIR}/bin"
add_to_rc() {
  local rc="$1"
  [ -f "$rc" ] && grep -qF "$BIN_DIR" "$rc" 2>/dev/null && return
  printf '\n# Added by MOSAICX setup\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >>"$rc"
  echo "ok updated $rc"
}
case "$(basename "${SHELL:-/bin/bash}")" in
  zsh)  add_to_rc "${HOME}/.zshrc" ;;
  fish) mkdir -p "${HOME}/.config/fish"
        rc="${HOME}/.config/fish/config.fish"
        if ! grep -qF "$BIN_DIR" "$rc" 2>/dev/null; then
          printf '\n# Added by MOSAICX setup\nfish_add_path %s\n' "$BIN_DIR" >>"$rc"
          echo "ok updated $rc"
        fi ;;
  *)    add_to_rc "${HOME}/.bashrc" ;;
esac
export PATH="${BIN_DIR}:${PATH}"
# Hand off to mosaicx setup
# shellcheck disable=SC2086
"${BIN_DIR}/mosaicx" setup $FULL $NONINTERACTIVE
