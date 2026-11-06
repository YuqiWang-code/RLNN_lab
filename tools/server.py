#!/usr/bin/env python3
"""Inspect the configured server and synchronize sources or small output files."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shlex
import stat
import subprocess

import paramiko

ROOT = Path(__file__).resolve().parents[1]
SMALL_OUTPUTS = {'.log', '.csv', '.json', '.jsonl', '.txt', '.md', '.yaml', '.yml'}
STATUS = """date -Is
hostname
uname -a
uptime
lscpu | head -25
free -h
df -h / /home
nvidia-smi
command -v python3
command -v g++
command -v make
"""


def mkdir(sftp, remote):
    try:
        sftp.stat(remote)
    except FileNotFoundError:
        mkdir(sftp, posixpath.dirname(remote))
        sftp.mkdir(remote)


def push(sftp, remote):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    count = 0
    for name in names:
        if not name or PurePosixPath(name).parts[0] in {'.vscode', 'outputs', 'tmp'}:
            continue
        source = ROOT / name
        if not source.is_file():
            continue
        target = posixpath.join(remote, name)
        mkdir(sftp, posixpath.dirname(target))
        local_stat = source.stat()
        try:
            remote_stat = sftp.stat(target)
        except FileNotFoundError:
            remote_stat = None
        if remote_stat and remote_stat.st_size == local_stat.st_size and int(remote_stat.st_mtime) == int(local_stat.st_mtime):
            continue
        sftp.put(str(source), target)
        sftp.utime(target, (int(local_stat.st_atime), int(local_stat.st_mtime)))
        if source.suffix == '.sh':
            sftp.chmod(target, 0o755)
        count += 1
    for folder in ('outputs', 'tmp', 'docs/temporary', 'models/ckpts'):
        mkdir(sftp, posixpath.join(remote, folder))
    print(f'Uploaded {count} files to {remote}')


def pull(sftp, remote, destination):
    count = 0
    for entry in sftp.listdir_attr(remote):
        source = posixpath.join(remote, entry.filename)
        target = destination / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            if entry.filename not in {'checkpoints', 'wandb', '__pycache__'}:
                count += pull(sftp, source, target)
        elif stat.S_ISREG(entry.st_mode) and target.suffix.lower() in SMALL_OUTPUTS and entry.st_size <= 50 * 1024 * 1024:
            if target.exists() and target.stat().st_size == entry.st_size and int(target.stat().st_mtime) == int(entry.st_mtime):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(source, str(target))
            os.utime(target, (entry.st_atime, entry.st_mtime))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'push', 'pull'])
    parser.add_argument('--save', type=Path, help='save status output as UTF-8 text')
    args = parser.parse_args()
    config = json.loads((ROOT / '.vscode/sftp.json').read_text(encoding='utf-8-sig'))
    remote = config['remotePath'].rstrip('/')
    with paramiko.SSHClient() as client:
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(config['host'], port=config.get('port', 22), username=config['username'],
                       password=config.get('password'), timeout=20, look_for_keys=False, allow_agent=False)
        if args.action == 'status':
            _, stdout, stderr = client.exec_command(STATUS + '\nls -la ' + shlex.quote(remote), timeout=40)
            report = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(report, encoding='utf-8')
            print(report)
        else:
            with client.open_sftp() as sftp:
                if args.action == 'push':
                    push(sftp, remote)
                else:
                    print(f'Downloaded {pull(sftp, remote + "/outputs", ROOT / "outputs")} output files')


if __name__ == '__main__':
    main()
