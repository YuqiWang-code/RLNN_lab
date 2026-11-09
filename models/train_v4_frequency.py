"""Train state adapters on matched trajectories from a frozen RLNN teacher."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from main import build_parser, set_seed
from models import GCN, edge2adj
from solvers import RLNN


class V4TrainingRLNN(RLNN):
    def print_log(self, logger):
        model = self.accelerator.unwrap_model(self.model)
        adapter = getattr(model, 'adapter', None)
        if adapter is not None:
            logger['alpha_low'] = float(adapter.alpha_low.detach())
            logger['alpha_high'] = float(adapter.alpha_high.detach())
            for name in ('low', 'high', 'gates'):
                gradients = [p.grad.detach().square().sum() for p in getattr(adapter, name).parameters() if p.grad is not None]
                logger['last_grad_' + name] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
        super().print_log(logger)


class MatchedStateRLNN(V4TrainingRLNN):
    def __init__(self, args):
        super().__init__(args)
        self.teacher = GCN(1, args.num_h, args.num_l, 1)
        self.teacher.load_state_dict(torch.load(args.warm_start, map_location='cpu', weights_only=True))
        self.teacher.to(device=self.accelerator.device, dtype=torch.float32).eval()
        self.teacher.requires_grad_(False)

    @torch.no_grad()
    def generate_data(self, graph, A=None, normalized=None):
        x = torch.randint(0, 2, (graph.num_nodes, self.num_k),
                          device=self.accelerator.device, dtype=torch.float32)
        adj = edge2adj(graph.edge_index, graph.num_nodes)
        states = []
        for _ in range(self.num_tp):
            states.append(x.clone())
            logits = self.teacher(x.unsqueeze(-1), graph.edge_index, adj).squeeze(-1)
            x = torch.where(torch.rand_like(x) < logits.sigmoid(), 1 - x, x)
        return torch.cat(states, dim=1).to(self.dtype)


def main():
    parser = build_parser()
    parser.add_argument('--result_dir', type=Path, required=True)
    parser.add_argument('--state_source', choices=('teacher', 'on_policy'), default='teacher')
    args = parser.parse_args()
    if not args.warm_start or args.resume:
        parser.error('v4 screening starts from --warm_start without --resume')
    args.do_train = True
    run = args.result_dir
    run.mkdir(parents=True, exist_ok=False)
    args.save_dir = str(run / 'checkpoints')
    Path(args.save_dir).mkdir()
    config = vars(args).copy()
    config['result_dir'] = str(config['result_dir'])
    record = dict(version='v4_frequency', mode='train', status='running',
                  seed=args.seed, frequency=args.frequency, config=config,
                  started_at=datetime.now(timezone.utc).isoformat())
    def save():
        (run / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    start = time.perf_counter()
    try:
        set_seed(args.seed)
        solver = (MatchedStateRLNN if args.state_source == 'teacher' else V4TrainingRLNN)(args)
        solver.prepare()
        # Model construction consumes different amounts of randomness across arms.
        set_seed(20261 + args.seed)
        record['params'] = sum(p.numel() for p in solver.model.parameters())
        record['adapter_params'] = sum(p.numel() for n, p in solver.model.named_parameters() if n.startswith('adapter.'))
        record['device'] = {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                            'cuda': torch.version.cuda, 'training_stream': 20261 + args.seed}
        save()
        solver.train(args.save_dir)
        state = torch.load(Path(args.save_dir) / 'latest_training.pt', map_location='cpu', weights_only=False)
        history = state['history']
        with (run / 'training.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        record.update(status='completed', completed_epochs=state['epoch'],
                      updates=history[-1]['Iteration'], best_validation_size=-state['best_eval'],
                      peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start)
        save()


if __name__ == '__main__':
    main()
