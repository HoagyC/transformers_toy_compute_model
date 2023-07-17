import json
import os
import subprocess
import sys
from typing import Optional
from pathlib import Path

VAST_NUM = 4
VAST_PORT = 37182
SSH_DIRECTORY = "sparse_coding"
dest_addr = f"root@ssh{VAST_NUM}.vast.ai"
SSH_PYTHON = "/opt/conda/bin/python"

def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e "ssh -p {VAST_PORT}" . {dest_addr}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()