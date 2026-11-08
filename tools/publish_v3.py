"""Synchronize completed v3 experiment records and the research workbook."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v3'
NODE=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'


def command(*arguments,capture=False):
    result=subprocess.run(list(map(str,arguments)),cwd=ROOT,check=True,text=True,encoding='utf-8',capture_output=capture)
    return result.stdout.strip() if capture else None


def publish():
    command(sys.executable,'tools/server.py','pull','--version','v3')
    paths=[]
    for file in sorted(OUT.rglob('run.json')):
        if any(part.startswith('smoke_') for part in file.relative_to(OUT).parts):
            continue
        record=json.loads(file.read_text(encoding='utf-8'))
        if record.get('status') in ('completed','skipped'):
            paths.append(file.parent.relative_to(ROOT).as_posix())
    for file in sorted(OUT.glob('*decision*.json')):
        paths.append(file.relative_to(ROOT).as_posix())
    if (OUT/'budget_selection.json').is_file(): paths.append('outputs/v3/budget_selection.json')
    done=(OUT/'queue/v3.done.txt').is_file()
    if done:
        paths += [file.relative_to(ROOT).as_posix() for file in (OUT/'queue').glob('*.done.txt')]
    changed=bool(paths and command('git','status','--porcelain','--',*paths,capture=True))
    if changed:
        command(sys.executable,'tools/summarize_v3.py')
        command(NODE,'tools/update_v3_workbook.mjs')
        paths += ['outputs/v3/summary.json','outputs/v3/summary.txt','docs/实验结果.xlsx']
        if done:
            snapshot=command(sys.executable,'tools/snapshot.py','--version','v3',capture=True).splitlines()[-1]
            paths.append(snapshot)
        command('git','add','--',*paths)
        command('git','commit','--only','-m','Record completed v3 MIS experiments','--',*paths)
    if command('git','rev-list','origin/main..HEAD',capture=True):
        command('git','push','origin','main')
        command(sys.executable,'tools/server.py','push')
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=ROOT.as_posix())
    while True:
        try:
            if publish() or not args.watch: return
        except (OSError,subprocess.CalledProcessError) as error:
            print(f'Publisher will retry: {error}',flush=True)
            if not args.watch: raise
        time.sleep(180)


if __name__=='__main__':
    main()
