#!/usr/bin/env python3
"""Collect and publish the completed v000 RLNN baseline from the server."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import paramiko

ROOT=Path(__file__).resolve().parents[1]
VERSION='v000_rlnn_baseline'
OUT=ROOT/'outputs'/VERSION
NODE=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
STATUS=ROOT/'tmp/baseline_completion.json'


def state(value, **extra):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(dict(state=value,at=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n',encoding='utf-8')


def command(*args):
    result=subprocess.run([str(x) for x in args],cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
    print(result.stdout,flush=True)
    return result.stdout.strip()


def pipeline_result():
    config=json.loads((ROOT/'.vscode/sftp.json').read_text(encoding='utf-8-sig'))
    remote=config['remotePath'].rstrip('/')
    if remote!='/home/hzeng/project/ZH/RLNN_lab' or not config['deploymentConfirmed']:
        raise ValueError('RLNN deployment is not configured')
    with paramiko.SSHClient() as client:
        client.load_system_host_keys(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(config['host'],port=config.get('port',22),username=config['username'],password=config.get('password'),timeout=20,look_for_keys=False,allow_agent=False)
        _,stdout,stderr=client.exec_command(f'if [ -f {remote}/outputs/{VERSION}/pipeline_g13_cache_w8/exit_code.txt ]; then cat {remote}/outputs/{VERSION}/pipeline_g13_cache_w8/exit_code.txt; else echo running; fi',timeout=20)
        return stdout.read().decode().strip()


def metrics(run):
    with (OUT/run/'metrics/metrics.csv').open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def finalize():
    command(sys.executable,'tools/server.py','pull')
    train_name='rb_train_g13_cache_w8_gpu1_seed0'; test_name='rb_test_g13_cache_w8_gpu1_seed0'
    train=json.loads((OUT/train_name/'run.json').read_text())
    test=json.loads((OUT/test_name/'run.json').read_text())
    if train['status']!='completed' or test['status']!='completed':
        raise ValueError('Training and evaluation must both be completed')
    rows=metrics(train_name)
    epochs=[row for row in rows if row.get('val/raw_obj')]
    if len(epochs)!=50 or max(int(row['epoch']) for row in epochs)!=50:
        raise ValueError('The full 50-epoch training log is required')
    if int(epochs[-1].get('step') or 0) != 3850:
        raise ValueError('Expected 3850 optimizer updates for 13-graph batching')
    best=min(epochs,key=lambda row:float(row['val/raw_obj']))
    tested=next(row for row in metrics(test_name) if row.get('test/raw_obj'))
    manifest=json.loads((OUT/'data/manifest.json').read_text())
    if [manifest['splits'][s]['count'] for s in ['train','val','test']] != [1000,100,500]:
        raise ValueError('Expected 1000/100/500 graph splits')
    command(NODE,'tools/collect_results.mjs')
    command(NODE,'tools/collect_results.mjs','--kind','smoke')
    score=float(tested['test/independent_set_size']); seconds=float(tested['test/time'])
    report=f'''# RLNN 单卡基线结果

MIS RB200–300，版本 `{VERSION}`，seed 0，单张 RTX 5090（GPU 1）。训练与独立测试均已完成。

| 项目 | 实测值 |
| --- | --- |
| 训练 / 验证 / 测试图数 | 1000 / 100 / 500 |
| 训练 epoch | 50 |
| 最佳验证 epoch | {int(best['epoch'])} |
| 最佳验证独立集大小均值 | {-float(best['val/raw_obj']):.6f} |
| 测试独立集大小均值 | {score:.6f} |
| 测试原始负目标 | {float(tested['test/raw_obj']):.6f} |
| 测试每图平均采样与解码时间（秒） | {seconds:.6f} |
| 本段续训墙钟时间（小时） | {train['wall_time_s']/3600:.4f} |
| 测试进程墙钟时间（秒） | {test['wall_time_s']:.6f} |

数据使用预转换 CPU 张量缓存、8 个常驻 worker 和 pinned memory。训练从旧运行第 22 轮最佳权重继续到总第 50 轮；该起点缺少 Adam 和随机数状态，二者已重置并记录在 run.json。后续每轮保存完整训练状态。

训练配置为 RLNN / erdoes、5 层 GCN、隐藏维 128、13 图合批、状态 batch 1024（每图实际 1000 个状态，每次更新通常 13000 状态）、Adam lr 0.0001、K=20、T=100、T′=50、d=5、β=1.02、λ=0.5；训练 float32、测试 fp16。使用验证集选择 best.pt，测试阶段加载该权重。

数据由官方 RB 生成器独立生成，train/val/test 种子起点为 0/100000/200000，实际节点数范围均为 200–264。完整统计见[数据清单](../outputs/{VERSION}/data/manifest.json)。

原始日志、命令、设备、配置、CSV 位于[训练目录](../outputs/{VERSION}/{train_name}/)与[测试目录](../outputs/{VERSION}/{test_name}/)。工作簿保留逐 epoch 原始负目标、派生独立集大小、平滑训练损失和时间。8 节点冒烟与 fp16 冒烟结果单独保存在冒烟工作簿。

## 已知差异

本结果是 RLNN 单卡多图合批训练。Epoch 1–22 来自原运行，23–50 来自张量缓存续训；初始续训仅恢复权重，Adam 与随机状态重置。报告墙钟时间为本段续训耗时。每图采样预算保持不变，损失按图平均，每 epoch 77 次更新，50 epoch 共 3850 次更新。训练粒子数、状态 batch、多卡预算及解码与论文的差异见[RLNN基线](RLNN基线.md)。测试图是本项目生成集，未与作者测试图对齐；没有可靠参考最优值，不计算 gap。每图时间沿用上游统计，未显式 CUDA 同步；与完整训练墙钟时间分别记录。
'''
    (ROOT/'docs/实验报告.md').write_text(report,encoding='utf-8',newline='\n')
    (ROOT/'docs/项目现状.md').write_text(f'''# 项目现状

当前版本 `{VERSION}`，MIS RB200–300，单张 RTX 5090（GPU 1）。

- 源码迁移、服务器独立部署、环境安装已完成。
- GPU 冒烟训练、checkpoint 加载、解码和 fp16 推理已通过。
- RB train/val/test 1000/100/500 图已准备，数据留在服务器。
- 50 epoch 单卡训练与 500 图独立测试已完成。
- 测试独立集大小均值为 {score:.6f}，逐项结果见[实验报告](实验报告.md)。
- 真实指标已整理到正式及冒烟工作簿；权重留在服务器。

部署路径 `/home/hzeng/project/ZH/RLNN_lab`，环境 `/home/hzeng/envs/rlnn`。本地凭据不入库，自动保存上传关闭。配置与论文差异见[RLNN基线](RLNN基线.md)。
''',encoding='utf-8',newline='\n')
    p=ROOT/'README.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('张量缓存与 8 worker 的续训正在运行，目标为总第 50 epoch；独立测试尚未完成。',f'完整 50 epoch 单卡训练与独立测试已完成，测试独立集大小均值为 {score:.6f}。详见[实验报告](docs/实验报告.md)。')
    p.write_text(text,encoding='utf-8',newline='\n')
    p=ROOT/'docs/RLNN基线.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('GPU 环境、RB 数据、图缓存与续训检查已完成；缓存和 8 worker 续训运行中，独立测试尚未完成。','GPU 环境、RB 数据、端到端冒烟、50 epoch 正式训练与 500 图测试均已完成，结果见[实验报告](实验报告.md)。')
    p.write_text(text,encoding='utf-8',newline='\n')
    p=ROOT/'docs/验证记录.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('正式训练运行中，尚无完整训练与独立测试成绩。','50 epoch 正式训练与独立的 500 图测试已完成，成绩见实验报告。')
    text=text.replace('50 epoch 重训已启动，完整测试尚未完成。','50 epoch 重训与 500 图完整测试已完成。')
    p.write_text(text,encoding='utf-8',newline='\n')
    p=ROOT/'docs/已归档的想法和文件/当前想法.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('单卡基线正在 GPU 1 训练。','单卡基线已在 GPU 1 完成训练与独立测试。')
    text=text.replace('完成 50 epoch 与 500 图独立测试，保留真实损失、验证目标、测试目标和计时。','依据已完成的 50 epoch 与 500 图测试，设计后续多种子与受控对照。')
    p.write_text(text,encoding='utf-8',newline='\n')
    snapshot=command(sys.executable,'tools/snapshot.py','--version',VERSION).splitlines()[-1]
    command(sys.executable,'tools/server.py','push')
    command('git','add','--','outputs/'+VERSION,'docs/实验结果.xlsx','docs/冒烟结果.xlsx','docs/实验报告.md','docs/项目现状.md','docs/RLNN基线.md','docs/验证记录.md','docs/已归档的想法和文件/当前想法.md','README.md',snapshot)
    command('git','commit','-m','Record RLNN GPU 1 baseline training and evaluation results')
    command('git','push','origin','main')
    state('completed',test_independent_set_size=score,snapshot=snapshot,commit=command('git','rev-parse','HEAD'))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--watch',action='store_true'); args=parser.parse_args()
    state('waiting')
    while True:
        try: result=pipeline_result()
        except (OSError,paramiko.SSHException) as error:
            if not args.watch: raise
            state('waiting',connection_error=str(error)); time.sleep(60); continue
        if result=='running':
            if not args.watch: raise RuntimeError('Server pipeline is still running')
            time.sleep(60); continue
        if result!='0':
            command(sys.executable,'tools/server.py','pull')
            raise RuntimeError(f'Server pipeline exited with {result}')
        break
    state('finalizing'); finalize()


if __name__=='__main__':
    try: main()
    except Exception as error:
        state('failed',error=str(error)); raise
