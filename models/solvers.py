import os
import random
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW, Adam
from models import GCN, MLP, edge2adj
from mis_v1 import MISStateGCN, adjacency, search
from archived_frequency.mis_v2_frequency import SEGF, SEGF_MODES, frequency_topology
from archived_frequency.mis_v4_frequency import V4FrequencyGCN, V4_MODES, v4_topology
from archived_frequency.mis_v5_frequency import V5DAFRGCN, V5_MODES, v5_topology
from archived_frequency.mis_v6_frequency import V6SCGCN, V6_MODES, v6_topology, polarization_bonus
from mis_v7_context import STATIC_MODES, V7StaticContextGCN
from mis_structure import STRUCTURE_MODES, StructureGCN, structure_topology
from v7_structure import v7_topology
from torch_scatter import scatter_min, scatter_max, scatter_add
from torch_geometric.utils import add_remaining_self_loops
from accelerate import Accelerator
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
from utils import collate_fn, build_batch
import torch.nn.functional as F
import importlib
from pathlib import Path
import scipy.sparse as sp
from scipy.optimize import minimize

def erdoes_graph_batch_loss(problem, A, b, x, prob, graph_index, num_graphs, num_d, lambd, *, terms=False):
    flipped = (1-prob)*x + prob*(1-x)
    energy, _ = problem.energy_func(A, b, flipped, False)
    expected_flips = scatter_add(prob, graph_index, dim=0, dim_size=num_graphs)
    expected_energy = energy.mean()/num_graphs
    distance = ((expected_flips-num_d)**2).mean()
    loss = expected_energy + lambd*distance
    if terms:
        return loss, torch.stack((expected_energy.detach(), distance.detach(), expected_flips.detach().mean()))
    return loss


class MetaSolver(object):
    def __init__(self, args):
        self.num_t = args.num_t
        self.num_k = args.num_k
        self.model = None
        self.optimizer = None
        self.skip_decode = args.skip_decode
        self.mixed_precision = args.mixed_precision
        self.model = None
        self.optimizer = None
        
        module = importlib.import_module('.'.join(['problems' ,args.problem]))
        dataset = getattr(module, 'Dataset')
        problem = getattr(module, 'Problem')
        self.problem = problem(args.beta)
        
        if args.mixed_precision == 'no':
            self.dtype = torch.float
        elif args.mixed_precision == 'fp16':
            self.dtype = torch.float16
        elif args.mixed_precision == 'bf16':  
            # Note bf16 does not support kth value
            self.dtype = torch.bfloat16
        else: 
            raise NotImplementedError
        
        loader_options = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        if args.do_train:
            self.epochs = args.epochs
            train_dataset = dataset(args.train_path, self.dtype, args.train_samples)
            valid_dataset = dataset(args.valid_path, self.dtype, args.valid_samples)
            self.train_loader = DataLoader(train_dataset, batch_size=args.graph_batch_size, **loader_options, shuffle=True, collate_fn=collate_fn) 
            self.valid_loader = DataLoader(valid_dataset, batch_size=1, **loader_options, shuffle=False, collate_fn=collate_fn)
            self.test_loader = None
        else:
            test_dataset = dataset(args.test_path, self.dtype, args.test_samples)
            self.test_loader = DataLoader(test_dataset, batch_size=1, **loader_options, shuffle=False, collate_fn=collate_fn)
            self.train_loader = None
            self.valid_loader = None
            
        accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.accelerator = accelerator
            
    def prepare(self):
        self.model, self.optimzier, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)
        
    def print_log(self, logger):
        log = []
        for k, v in logger.items():
            log.extend([k + ':', str(v)])
        log = ' '.join(log)
        self.accelerator.print(log)

    def load_ckpt(self, ckpt):
        self.model.load_state_dict(ckpt)

    def save_ckpt(self, save_file):
        torch.save(self.accelerator.get_state_dict(self.model), save_file)
    
    def train_batch(graphs):
        pass

    def evaluate_single(graph):
        pass
    
    def train(self, save_dir):
        best_eval = 0
        num_iter = 0
        logger = {'Epoch': 0, 'Iteration': 0, 'loss': None, 'valid_obj': None, 'valid_time': None}    
        best_eval = np.inf
        iterations = 0
        patience = 10
        wait = 0
        for epoch in range(self.epochs):
            logger['Epoch'] = epoch+1
            self.model.train()
            for index, graphs in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=not self.accelerator.is_local_main_process):  
                logger['Iteration'] = index+1
                iterations += 1
                loss = self.train_batch(graphs)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()   
                if logger['loss'] is None:
                    logger['loss'] = loss.item()
                else:
                    logger['loss'] = 0.95*logger['loss'] + 0.05*loss.item()
                
            self.accelerator.wait_for_everyone() 
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            logger['valid_obj'] = obj_result
            logger['valid_time'] = time_result
            self.print_log(logger)
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
                wait = 0
            else:
                wait += 1
            if wait == patience:
                self.accelerator.print('Early stop')
                break

    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        assert mode in ['valid', 'test']
        if mode == 'valid':
            dataloader = self.valid_loader
        else:
            dataloader = self.test_loader
        if self.model is not None:
            self.model.eval()
        all_ids = []
        all_time = []   
        all_eval_set_size = []
        loss_trackers = []
        for batch_idx, graphs in tqdm(enumerate(dataloader), total=len(dataloader), disable=not self.accelerator.is_local_main_process):
            for graph in graphs: 
                st_time = time.time()
                eval_set_size = self.evaluate_single(graph)
                end_time = time.time()
                all_ids.append(torch.tensor(graph.idx, device=self.accelerator.device))
                all_time.append(torch.tensor(end_time-st_time, device=self.accelerator.device))
                all_eval_set_size.append(eval_set_size)

        all_time = torch.stack(all_time)
        all_time = self.accelerator.gather(all_time)
                               
        all_eval_set_size = torch.stack(all_eval_set_size)        
        all_eval_set_size = self.accelerator.gather(all_eval_set_size)

        obj_result = torch.mean(all_eval_set_size).item()
        time_result = torch.mean(all_time).item()
        
        if log_path is not None:
            if self.accelerator.is_local_main_process:
                with open(log_path, 'w') as f:
                    f.write('Avg. obj: %f\n'%obj_result)
                    f.write('Avg. time: %f\n'%time_result)
                #loss_trackers.dump(log_path.replace('txt', 'npy'))
                self.accelerator.print(obj_result, time_result)
        
        return obj_result, time_result


class RLSA(MetaSolver):
    def __init__(self, args):
        super(RLSA, self).__init__(args)
        self.num_d = args.num_d
        self.tau0 = args.tau0
        
    def evaluate_single(self, graph, *, return_solution=False, generator=None):
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device,
                          dtype=graph.edge_weight.dtype, generator=generator)
        A = torch.sparse_coo_tensor(
            graph.edge_index, 
            graph.edge_weight, 
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        energy, grad = self.problem.energy_func(A, graph.b, x, True)
        
        best_sol = x.clone()
        best_energy = energy.clone()
        
        for epoch in range(self.num_t):          
            tau = self.tau0*(1-epoch/self.num_t)
            delta = grad*(2*x-1)/2
            
            #The kth value method
            term2 = -torch.kthvalue(-delta, self.num_d, dim=0, keepdim=True).values
            flip_prob = torch.sigmoid((delta-term2)/tau)
            
            #The normalization method
            #flip_prob = torch.sigmoid(delta/tau)
            #flip_prob = flip_prob*self.num_d/flip_prob.sum(dim=0, keepdim=True)
            
            rr = torch.rand_like(x) if generator is None else torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = torch.where(rr<flip_prob, 1-x, x)
            
            energy, grad = self.problem.energy_func(A, graph.b, x, True if epoch < self.num_t-1 else False)
            best_sol = torch.where((energy<best_energy).unsqueeze(0).repeat(graph.num_nodes,1), x, best_sol)
            best_energy = torch.where(energy<best_energy, energy, best_energy)
               
        if return_solution:
            solutions = self.problem.decode_gradient_flip(best_sol, A, graph.b, return_solutions=True)
            return solutions[:, solutions.sum(dim=0).argmax()]
        if self.skip_decode:
            return best_energy.min()
        else:
            best_energy = self.problem.decode_func(best_sol, graph)
            return best_energy.min()
        

class RLNN(MetaSolver):
    def __init__(self, args):
        super(RLNN, self).__init__(args)
        
        self.frequency = getattr(args, 'frequency', 'none')
        self.segf = self.frequency in SEGF_MODES
        self.v4_frequency = self.frequency in V4_MODES
        self.v5_frequency = self.frequency in V5_MODES
        self.v6_frequency = self.frequency in V6_MODES
        self.context_mode = getattr(args, 'context_mode', 'none')
        self.v7_context = self.context_mode != 'none'
        self.structure_mode = getattr(args, 'structure_mode', 'none') or 'none'
        self.structure = self.structure_mode != 'none'
        self.structure_gain = False
        if self.structure and (self.context_mode != 'none' or self.frequency != 'none'):
            raise ValueError('Structural compensation cannot be combined with another encoder')
        if self.structure and args.problem != 'mis':
            raise ValueError('Structural compensation requires MIS')
        if args.do_train and self.context_mode in ('static_random', 'static_oracle'):
            raise ValueError('Random and oracle modes are inference interventions')
        if self.structure:
            self.model = StructureGCN(args.num_h, args.num_l, self.structure_mode, args.beta)
            self.structure_gain = self.model.needs_product
        elif self.v7_context:
            if args.problem != 'mis' or self.frequency != 'none':
                raise ValueError('Static context requires MIS without a legacy frequency adapter')
            self.model = V7StaticContextGCN(args.num_h, args.num_l, self.context_mode,
                                           args.v7_width, args.v7_d_state, args.v7_delta_limit,
                                           static_node_limit=args.static_node_limit, static_encoder=args.static_encoder,
                                           constant_limit=args.static_constant_limit, feature_dim=args.static_features,
                                           peak_normalize=args.static_peak_normalize,
                                           bias_seed=args.static_bias_seed, oracle_path=args.static_oracle_path,
                                           order_seed=args.v7_order_seed, views=args.v7_order_views)
            self.order_seed, self.order_views = args.v7_order_seed, args.v7_order_views
        elif self.frequency != 'none':
            if args.problem != 'mis':
                raise ValueError('State frequency encoders require MIS')
            model_type = (V6SCGCN if self.v6_frequency else V5DAFRGCN if self.v5_frequency else V4FrequencyGCN if self.v4_frequency
                          else SEGF if self.segf else MISStateGCN)
            self.model = model_type(args.num_h, args.num_l, args.beta, self.frequency)
        else:
            self.model = GCN(1, args.num_h, args.num_l, 1)
        self.model.reset_parameters()
        
        self.validation_seed = getattr(args, 'validation_seed', None)
        self.freeze_backbone_epochs = getattr(args, 'freeze_backbone_epochs', 0)
        self.finetune_lr = getattr(args, 'finetune_lr', None)
        warm_start = getattr(args, 'warm_start', None)
        if args.do_train:
            if warm_start and not args.resume:
                weights = torch.load(warm_start, map_location='cpu', weights_only=True)
                if self.v5_frequency and any(key.startswith('adapter.') for key in weights):
                    raise ValueError('v5 warm start requires a plain RLNN checkpoint')
                missing, extra = self.model.load_state_dict(weights, strict=not (self.segf or self.v4_frequency or self.v5_frequency or self.v7_context or self.structure))
                if extra or any(not key.startswith(('adapter.', 'compensation.')) for key in missing):
                    raise ValueError('Warm start must contain the original RLNN backbone')
            self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        else:
            state_dict = torch.load(
                os.path.join(args.save_dir, 'best.pt'), 
                weights_only=True, 
                map_location=self.accelerator.device
            )
            if self.context_mode in ('static_random', 'static_oracle'):
                missing, extra = self.model.load_state_dict(state_dict, strict=False)
                if any(not key.startswith('adapter.') for key in (*missing, *extra)):
                    raise ValueError('Static interventions require an RLNN backbone checkpoint')
            else:
                self.load_ckpt(state_dict)

        self.num_d = args.num_d
        self.lambd = args.lambd
        self.num_tp = args.num_tp
        self.num_kp = args.num_kp
        self.batch_size = args.batch_size
        self.training_config = vars(args).copy()
        self.resume_path = args.resume
        self.resume_state = None
        self.graph_batch_size = args.graph_batch_size
        self.loss = args.loss
        self.lambda_dir = getattr(args, 'lambda_dir', 0.03)
        self.lambda_ortho = getattr(args, 'lambda_ortho', 0.01)
        if args.do_train and self.v6_frequency and (self.graph_batch_size < 2 or self.loss != 'erdoes'):
            raise ValueError('v6 training requires multiple-graph MIS erdoes updates')
        if self.resume_path and self.graph_batch_size < 2:
            raise ValueError("Epoch resume requires graph_batch_size > 1")
        if self.graph_batch_size > 1 and self.loss != "erdoes":
            raise ValueError("Multiple-graph training requires erdoes loss")
        if args.do_train and (self.freeze_backbone_epochs or self.finetune_lr or self.validation_seed is not None):
            if self.graph_batch_size < 2 or args.problem != 'mis':
                raise ValueError("V2 training requires multiple-graph MIS erdoes updates")

    def prepare(self):
        super().prepare()
        if self.resume_path:
            self.resume_state = torch.load(self.resume_path, map_location='cpu', weights_only=False)
            self.load_ckpt(self.resume_state['model'])
            if self.resume_state['optimizer'] is not None:
                self.optimizer.load_state_dict(self.resume_state['optimizer'])
            rng = self.resume_state['rng']
            if rng is not None:
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                torch.set_rng_state(rng['torch'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng['cuda'])

    def save_training_state(self, save_dir, epoch, best_eval, logger, history):
        state = {'model': self.accelerator.get_state_dict(self.model), 'config': self.training_config, 'source_run': os.path.dirname(save_dir) if os.path.basename(save_dir)=='checkpoints' else save_dir,
            'optimizer': self.optimizer.state_dict(), 'best_model': torch.load(os.path.join(save_dir, 'best.pt'), map_location='cpu', weights_only=True), 'epoch': epoch,
            'best_eval': best_eval, 'logger': dict(logger), 'history': list(history),
            'resume_mode': 'training_state',
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        torch.save(state, os.path.join(save_dir, 'latest_training.pt'))

    def train(self, save_dir):
        if self.graph_batch_size > 1:
            return self.train_graph_batches(save_dir)
        best_eval = 0
        num_iter = 0
        logger = {'Epoch': 0, 'Iteration': 0, 'loss': None, 'valid_obj': None, 'valid_time': None}    
        best_eval = np.inf
        iterations = 0
        self.accelerator.wait_for_everyone() 

        self.print_log(logger)
        for epoch in range(self.epochs):
            logger['Epoch'] = epoch+1
            self.model.train()
            for index, graphs in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=not self.accelerator.is_local_main_process): 

                for graph in graphs:
                    A = adjacency(graph)
                    normalized = self.policy_topology(graph, A)
                    inputs = self.generate_data(graph, A, normalized)
                    num_samples = inputs.shape[1]

                    num_batch = int(np.ceil(num_samples/self.batch_size))
                    for i in range(num_batch):
                        x = inputs[:,i*self.batch_size:(i+1)*self.batch_size]
                        logits = self.flip_logits(x, graph, A, normalized)
                        prob = torch.sigmoid(logits)
                        if self.loss == 'erdoes':
                            x = (1-prob)*x + prob*(1-x)
                            energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            loss = energy.mean()+self.lambd*((torch.sum(prob,0)-self.num_d)**2).mean()
                        elif self.loss == 'reinforce':
                            m = Bernoulli(prob)
                            change = m.sample()
                            log_prob = m.log_prob(change).sum(0)
                            original_energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            x = (1-change)*x + change*(1-x)
                            energy, _ = self.problem.energy_func(A, graph.b, x, False)
                            reward = original_energy-energy
                            loss = -(log_prob*reward).mean() + self.lambd*((torch.sum(prob,0)-self.num_d)**2).mean()
                        else:
                            raise NotImplementedError
                        
                        self.optimizer.zero_grad()
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        if logger['loss'] is None:
                            logger['loss'] = loss.item()
                        else:
                            logger['loss'] = 0.95*logger['loss'] + 0.05*loss.item()
                                
                iterations += 1

            self.accelerator.wait_for_everyone() 
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            logger['valid_obj'] = obj_result
            logger['valid_time'] = time_result
            self.print_log(logger)
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
    
    def graph_batch_loss(self, graph, adjacency, topology, x, prob, *, terms):
        return erdoes_graph_batch_loss(self.problem, adjacency, graph.b, x, prob,
                                       graph.batch, len(graph.graph_ptr) - 1, self.num_d, self.lambd, terms=terms)

    def train_graph_batches(self, save_dir):
        best_eval = np.inf
        logger = {'Epoch': 0, 'Iteration': 0, 'loss': None, 'valid_obj': None, 'valid_time': None}
        start_epoch = 0
        history = []
        if self.resume_state is not None:
            start_epoch = self.resume_state['epoch']
            best_eval = self.resume_state['best_eval']
            logger = dict(self.resume_state['logger'])
            history = list(self.resume_state['history'])
            torch.save(self.resume_state['best_model'], os.path.join(save_dir, 'best.pt'))
        self.accelerator.wait_for_everyone()
        self.print_log(logger)
        for epoch in range(start_epoch, self.epochs):
            logger['Epoch'] = epoch+1
            self.training_epoch = epoch
            self.model.train()
            if self.freeze_backbone_epochs:
                for name, parameter in self.model.named_parameters():
                    parameter.requires_grad_(name.startswith('adapter.') or epoch >= self.freeze_backbone_epochs)
            if self.finetune_lr is not None:
                rate = self.finetune_lr if epoch >= max(1, self.freeze_backbone_epochs) else self.training_config['lr']
                for group in self.optimizer.param_groups:
                    group['lr'] = rate
            track_terms = self.validation_seed is not None
            if track_terms:
                epoch_terms = torch.zeros(3, device=self.accelerator.device)
                updates = 0
            for graphs in tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process):
                graph = build_batch(graphs)
                A = adjacency(graph)
                normalized = self.policy_topology(graph, A)
                inputs = self.generate_data(graph, A, normalized)
                for start in range(0, inputs.shape[1], self.batch_size):
                    x = inputs[:, start:start+self.batch_size]
                    if self.v6_frequency:
                        logits, counts, orthogonality = self.model(
                            x, graph.edge_index, A, normalized, regularize=True)
                    else:
                        product = (A @ x) if self.structure_gain else None
                        logits = self.flip_logits(x, graph, A, normalized, adjacency_product=product)
                    prob = logits.sigmoid()
                    loss = self.graph_batch_loss(graph, A, normalized, x, prob, terms=track_terms)
                    if track_terms:
                        loss, values = loss
                        epoch_terms += values
                        updates += 1
                    if self.v6_frequency:
                        if self.frequency != 'v6_noanti' and self.lambda_dir:
                            loss = loss - self.lambda_dir * polarization_bonus(
                                x, prob, counts, normalized, len(graphs))
                        if orthogonality is not None and self.lambda_ortho:
                            loss = loss + self.lambda_ortho * orthogonality.mean()
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    logger['Iteration'] += 1
                    logger['loss'] = loss.item() if logger['loss'] is None else 0.95*logger['loss']+0.05*loss.item()
            if track_terms:
                logger.update(zip(('train_energy', 'train_distance', 'train_flips'), (epoch_terms / updates).tolist()))
                if self.accelerator.device.type == 'cuda':
                    logger['train_peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
            self.accelerator.wait_for_everyone()
            with torch.no_grad():
                obj_result, time_result = self.evaluate('valid')
            logger['valid_obj'], logger['valid_time'] = obj_result, time_result
            self.print_log(logger)
            if obj_result < best_eval:
                best_eval = obj_result
                self.save_ckpt(os.path.join(save_dir, 'best.pt'))
            history.append(dict(logger))
            self.save_training_state(save_dir, epoch+1, best_eval, logger, history)

    def policy_topology(self, graph, adj):
        if self.structure:
            topology = structure_topology(graph.edge_index, edge2adj(graph.edge_index, graph.num_nodes),
                                          graph.num_nodes, getattr(graph, 'graph_ptr', None))
            if not self.model.training:
                model = self.accelerator.unwrap_model(self.model)
                if model.compensation.use_degree or model.compensation.use_gate or model.compensation.use_readout:
                    with torch.no_grad():
                        bias, scale, readout = model.compensation.prepare(topology)
                    topology = topology._replace(cached_bias=bias, cached_scale=scale, cached_readout=readout)
            return topology
        if self.v7_context:
            model = self.accelerator.unwrap_model(self.model)
            adapter = model.adapter
            timings = getattr(self, 'context_timings', None)
            if self.context_mode in STATIC_MODES:
                graph_id = (Path(self.test_loader.dataset.files[graph.idx]).stem
                            if self.context_mode == 'static_oracle' else None)
                topology = adapter.topology(graph, adj, timings=timings, graph_id=graph_id)
            else:
                topology = v7_topology(graph, adapter.normalize, self.order_seed, order=adapter.order,
                                       views=self.order_views, timings=timings)
            if not self.model.training:
                if timings is not None:
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                with torch.no_grad():
                    intervention = getattr(self, 'static_intervention', None)
                    bias = (intervention(adapter, topology, graph) if intervention is not None
                            else adapter.action_bias(topology, model.delta_limit)).to(self.dtype)
                if timings is not None:
                    stop.record()
                    timings['_context_events'] = start, stop
                topology = topology._replace(cached_action_bias=bias)
            return topology
        if self.v6_frequency:
            return v6_topology(graph.edge_index, adj, graph.num_nodes)
        if self.v5_frequency:
            return v5_topology(graph.edge_index, adj, graph.num_nodes)
        if self.v4_frequency:
            return v4_topology(graph.edge_index, adj, graph.num_nodes)
        if self.segf:
            return frequency_topology(graph.edge_index, adj, graph.num_nodes)
        return edge2adj(graph.edge_index, graph.num_nodes)

    def flip_logits(self, x, graph, adjacency, normalized=None, adjacency_product=None):
        if self.structure:
            return self.model(x, graph.edge_index, normalized.normalized, normalized, adjacency_product)
        if self.v7_context:
            return self.model(x, graph.edge_index, adjacency, normalized)
        if self.frequency == 'none':
            return self.model(x.unsqueeze(-1), graph.edge_index, normalized).squeeze(-1)
        return self.model(x, graph.edge_index, adjacency, normalized)

    @torch.no_grad()
    def generate_data(self, graph, A=None, normalized=None):
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=self.dtype)
        if A is None:
            A = adjacency(graph)
        if normalized is None:
            normalized = self.policy_topology(graph, A)
        data = []

        for epoch in range(self.num_tp):
            data.append(x.clone())
            
            product = (A @ x) if self.structure_gain else None
            delta = self.flip_logits(x, graph, A, normalized, adjacency_product=product)
            flip_prob = torch.sigmoid(delta)

            rr = torch.rand_like(x)
            x = torch.where(rr<flip_prob, 1-x, x)
            
        return torch.cat(data, dim=-1)
          
    def sample_states(self, graph, adjacency, *, generator=None, adjust=None, normalized=None, with_energy=False):
        x = torch.randint(0, 2, (graph.num_nodes, self.num_k), device=self.accelerator.device,
                          dtype=self.dtype, generator=generator)
        if normalized is None:
            normalized = self.policy_topology(graph, adjacency)
        for step in range(self.num_t + 1):
            product = (adjacency @ x) if (with_energy or self.structure_gain) else None
            if with_energy:
                yield x, (x * (self.problem.beta * product + graph.b)).sum(dim=0)
            else:
                yield x
            if step == self.num_t:
                break
            if with_energy:
                logits = self.model(x, graph.edge_index, adjacency, normalized, adjacency_product=product)
            else:
                logits = self.flip_logits(x, graph, adjacency, normalized, adjacency_product=product)
            if adjust is not None:
                logits = adjust(x, logits)
            rr = torch.rand_like(x) if generator is None else torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = torch.where(rr < torch.sigmoid(logits), 1 - x, x)

    def evaluate_single(self, graph, *, decode=None):
        if self.validation_seed is not None:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.validation_seed + graph.idx)
            solution, _ = search(self, graph, generator)
            return -solution.sum()
        x = torch.randint(0,2, (graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=self.dtype)
        A = torch.sparse_coo_tensor(
            graph.edge_index, 
            graph.edge_weight, 
            torch.Size((graph.num_nodes, graph.num_nodes))
        ).to_sparse_csr()
        energy, _ = self.problem.energy_func(A, graph.b, x, False)
        
        best_sol = x.clone()
        best_energy = energy.clone()
        topology = self.policy_topology(graph, A) if (self.v7_context or self.structure) else None
        
        for epoch in range(self.num_t):          
            product = (A @ x) if self.structure_gain else None
            delta = self.flip_logits(x, graph, A, topology, adjacency_product=product)
            flip_prob = torch.sigmoid(delta)

            rr = torch.rand_like(x)
            x = torch.where(rr<flip_prob, 1-x, x)

            energy, _ = self.problem.energy_func(A, graph.b, x, False)        
            best_sol = torch.where((energy<best_energy).unsqueeze(0).repeat(graph.num_nodes,1), x, best_sol)
            best_energy = torch.where(energy<best_energy, energy, best_energy)

        if self.skip_decode:
            return best_energy.min()
        if decode is not None:
            return decode(best_sol, A, graph.b).min()
        return self.problem.decode_func(best_sol, graph).min()
