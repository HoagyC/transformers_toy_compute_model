import json
import os
import subprocess
import sys
from typing import Optional
from pathlib import Path

SSH_DIRECTORY = "toy_test"
DEST_ADDR = "mchorse@198.176.97.26"
SSH_PYTHON = "/opt/conda/bin/python"


def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e ssh  . {DEST_ADDR}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()
