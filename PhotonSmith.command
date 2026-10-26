#!/bin/zsh
# Local launcher for the source checkout. A fresh machine needs README setup.
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Create the Python environment using README.md, then launch again.'
  exit 1
fi
exec .venv/bin/python run.py "$@"
