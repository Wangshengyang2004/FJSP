from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action, select_action2
from models.PPO_Actor1 import Job_Actor, Mch_Actor
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from epsGreedyForMch import PredictMch
from uniform_instance import FJSPDataset
from FJSP_Env import FJSP
import os
import platform
from utils.device_utils import get_best_device
import gc
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_gpu_with_most_memory():
    """Get the GPU device with the most available memory."""
    if not torch.cuda.is_available():
        return get_best_device()
        
    # Get the number of available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return get_best_device()
        
    # Find GPU with most free memory
    max_free_memory = 0
    selected_gpu = 0
    
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            selected_gpu = i
            
    print(f"\nSelected GPU {selected_gpu} with {max_free_memory/1024**3:.2f} GB free memory")
    return torch.device(f"cuda:{selected_gpu}")

# Configure PyTorch memory management
torch.cuda.empty_cache()
if torch.cuda.is_available():
    # Enable memory efficient features
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Get the best GPU device
device = get_gpu_with_most_memory()
print(f"Using device: {device}")

# Enable gradient checkpointing for memory efficiency
def enable_gradient_checkpointing(model):
    if hasattr(model, 'encoder'):
        model.encoder.gradient_checkpointing = True
    if hasattr(model, 'feature_extract'):
        model.feature_extract.gradient_checkpointing = True

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.mask_mch = []
        self.first_task = []
        self.pre_task = []
        self.action = []
        self.mch = []
        self.dur = []
        self.mch_time = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.mask_mch[:]
        del self.first_task[:]
        del self.pre_task[:]
        del self.action[:]
        del self.mch[:]
        del self.dur[:]
        del self.mch_time[:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def initWeights(net, scheme='orthogonal'):

   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            nn.init.orthogonal_(e)

      elif scheme == 'normal':
         nn.init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         nn.init.xavier_normal(e)
def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs
class PPO:
    def __init__(self, rank, world_size, lr, gamma, k_epochs, eps_clip, n_j, n_m, num_layers, 
                 neighbor_pooling_type, input_dim, hidden_dim, num_mlp_layers_feature_extract,
                 num_mlp_layers_actor, hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
        # Enable memory efficient features
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        # Initialize models
        self.policy_job = Job_Actor(n_j=n_j, n_m=n_m, num_layers=num_layers,
                                  learn_eps=False, neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=self.device)
        
        self.policy_mch = Mch_Actor(n_j=n_j, n_m=n_m, num_layers=num_layers,
                                   learn_eps=False, neighbor_pooling_type=neighbor_pooling_type,
                                   input_dim=input_dim, hidden_dim=hidden_dim,
                                   num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                   device=self.device)

        # Move models to GPU and wrap with DDP
        self.policy_job = self.policy_job.to(self.device)
        self.policy_mch = self.policy_mch.to(self.device)
        
        self.policy_job = DDP(self.policy_job, device_ids=[rank], find_unused_parameters=True)
        self.policy_mch = DDP(self.policy_mch, device_ids=[rank], find_unused_parameters=True)
        
        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)

        # Initialize optimizers and schedulers
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)
        
        self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        
        self.MSE = nn.MSELoss()
        
        # Update GradScaler initialization
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Enable gradient checkpointing
        enable_gradient_checkpointing(self.policy_job)
        enable_gradient_checkpointing(self.policy_mch)

    def update(self,  memories, epoch):
        '''self.policy_job.train()
        self.policy_mch.train()'''
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []

        for i in range(configs.batch_size):
            rewards = []

            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.r_mb[0][i]).tolist()),
                                           reversed(memories.done_mb[0][i].tolist())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=self.device)

            job_log_prob = []
            mch_log_prob = []
            val = []
            mch_a =None
            last_hh = None
            entropies = []
            job_entropy = []
            mch_entropies = []
            job_scheduler = LambdaLR(self.job_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            mch_scheduler = LambdaLR(self.mch_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            env_mask_mch = memories.mask_mch[0]
            env_dur = memories.dur[0]
            first_task = memories.first_task[0]
            pool=None
            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i]
                env_adj = memories.adj_mb[i]
                env_candidate = memories.candidate_mb[i]
                env_mask = memories.mask_mb[i]


                a_index = memories.a_mb[i]
                env_mch_time = memories.mch_time[i]

                old_action = memories.action[i]
                old_mch = memories.mch[i]

                a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(x=env_fea,
                                                                                           graph_pool=g_pool_step,
                                                                                           padded_nei=None,
                                                                                           adj=env_adj,
                                                                                           candidate=env_candidate
                                                                                           , mask=env_mask
                                                                                           , mask_mch=env_mask_mch
                                                                                           , dur=env_dur
                                                                                           , a_index=a_index
                                                                                           , old_action=old_action
                                                                                           ,mch_pool=pool
                                                                                           , old_policy=False
                                                                                           )
                pi_mch,pool = self.policy_mch(action_node, hx, mask_mch_action, env_mch_time,mch_a,last_hh,policy=True)
                val.append(v)
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()

                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)
                # entropies.append((mch_entropy+a_entropy))

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)

            job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
                                                                                                     0).permute(1, 0)
            mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
                                                                                                     0).permute(1, 0)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            job_v_loss_sum = 0
            mch_loss_sum = 0
            mch_v_loss_sum = 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach()
                advantages = adv_normalize(advantages)
                
                # Use gradient checkpointing for forward passes
                job_surr1 = checkpoint(lambda x, y: x * y, job_ratios, advantages)
                job_surr2 = checkpoint(lambda x, y: torch.clamp(x, 1 - self.eps_clip, 1 + self.eps_clip) * y,
                                     job_ratios, advantages)
                
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                job_loss = -1*torch.min(job_surr1, job_surr2) + 0.5*job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss

                mch_surr1 = checkpoint(lambda x, y: x * y, mch_ratios, advantages)
                mch_surr2 = checkpoint(lambda x, y: torch.clamp(x, 1 - self.eps_clip, 1 + self.eps_clip) * y,
                                     mch_ratios, advantages)
                mch_loss = -1*torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
                mch_loss_sum += mch_loss

            # Optimize with mixed precision
            self.job_optimizer.zero_grad()
            self.scaler.scale(job_loss_sum.mean()).backward(retain_graph=True)
            self.scaler.step(self.job_optimizer)
            
            self.policy_old_job.load_state_dict(self.policy_job.state_dict())
            
            self.mch_optimizer.zero_grad()
            self.scaler.scale(mch_loss_sum.mean()).backward()
            self.scaler.step(self.mch_optimizer)
            self.scaler.update()
            
            self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
            
            if configs.decayflag:
                self.job_scheduler.step()
            if configs.decayflag:
                self.mch_scheduler.step()

            return job_loss_sum.mean().item(), mch_loss_sum.mean().item()

def train(rank, world_size):
    print(f"Running training on rank {rank}.")
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Enable memory efficient features
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Calculate local batch size
    total_batch_size = configs.batch_size
    local_batch_size = total_batch_size // world_size
    if local_batch_size == 0:
        local_batch_size = 1
    print(f"Rank {rank}: Local batch size = {local_batch_size}")
    
    # Create model and move it to GPU with DDP
    ppo = PPO(rank=rank, world_size=world_size,
              lr=configs.lr, gamma=configs.gamma, k_epochs=configs.k_epochs, 
              eps_clip=configs.eps_clip, n_j=configs.n_j, n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim, hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    
    # Create train and validation datasets with proper batch size
    train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins, 200)
    valid_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, 
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset,
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=False)
    
    # Create data loaders with distributed samplers and memory pinning
    train_loader = DataLoader(train_dataset, 
                            batch_size=local_batch_size,
                            sampler=train_sampler,
                            pin_memory=True,
                            num_workers=2,
                            persistent_workers=True)
    valid_loader = DataLoader(valid_dataset,
                            batch_size=local_batch_size,
                            sampler=valid_sampler,
                            pin_memory=True,
                            num_workers=2,
                            persistent_workers=True)
    
    # Initialize log list only on rank 0
    log = [] if rank == 0 else None
    
    record = float('inf')
    try:
        for epoch in range(1):
            train_sampler.set_epoch(epoch)
            memory = Memory()
            
            # Set models to training mode
            ppo.policy_old_job.train()
            ppo.policy_old_mch.train()
            
            times, losses, rewards2, critic_rewards = [], [], [], []
            start = time.time()
            
            costs = []
            losses, rewards, critic_loss = [], [], []
            
            for batch_idx, batch in enumerate(train_loader):
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                env = FJSP(configs.n_j, configs.n_m)
                data = batch.numpy()
                
                # Use gradient scaler context
                with torch.cuda.amp.autocast():
                    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
                    
                    job_log_prob = []
                    mch_log_prob = []
                    r_mb = []
                    done_mb = []
                    first_task = []
                    pretask = []
                    j = 0
                    mch_a = None
                    last_hh = None
                    pool = None
                    ep_rewards = - env.initQuality
                    
                    # Move data to device efficiently
                    env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device, non_blocking=True)
                    env_dur = torch.from_numpy(np.copy(dur)).float().to(device, non_blocking=True)

                    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                           batch_size=torch.Size(
                                               [local_batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                           n_nodes=configs.n_j * configs.n_m,
                                           device=device)
                    
                    while True:
                        env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
                        env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                        env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                        env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)

                        env_mask = torch.from_numpy(np.copy(mask)).to(device)
                        env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

                        action, a_idx, log_a, action_node, _, mask_mch_action, hx = ppo.policy_old_job(x=env_fea,
                                                                                                       graph_pool=g_pool_step,
                                                                                                       padded_nei=None,
                                                                                                       adj=env_adj,
                                                                                                       candidate=env_candidate,
                                                                                                       mask=env_mask,
                                                                                                       mask_mch=env_mask_mch,
                                                                                                       dur=env_dur,
                                                                                                       a_index=0,
                                                                                                       old_action=0,
                                                                                                       mch_pool=pool)

                        pi_mch, pool = ppo.policy_old_mch(action_node, hx, mask_mch_action, env_mch_time, mch_a, last_hh)
                        mch_a, log_mch = select_action2(pi_mch)
                        job_log_prob.append(log_a)
                        mch_log_prob.append(log_mch)

                        memory.mch.append(mch_a)
                        memory.pre_task.append(pretask)
                        memory.adj_mb.append(env_adj)
                        memory.fea_mb.append(env_fea)
                        memory.candidate_mb.append(env_candidate)
                        memory.action.append(deepcopy(action))
                        memory.mask_mb.append(env_mask)
                        memory.mch_time.append(env_mch_time)
                        memory.a_mb.append(a_idx)

                        adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(action.cpu().numpy(),
                                                                                                       mch_a)
                        ep_rewards += reward

                        r_mb.append(deepcopy(reward))
                        done_mb.append(deepcopy(done))

                        j += 1
                        if env.done():
                            break

                    memory.dur.append(env_dur)
                    memory.mask_mch.append(env_mask_mch)
                    memory.first_task.append(first_task)
                    memory.job_logprobs.append(job_log_prob)
                    memory.mch_logprobs.append(mch_log_prob)
                    memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0))
                    memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
                    
                    ep_rewards -= env.posRewards
                    loss, v_loss = ppo.update(memory, epoch)
                    memory.clear_memory()
                    mean_reward = np.mean(ep_rewards)
                    
                    if rank == 0:
                        log.append([batch_idx, mean_reward])
                        
                        if batch_idx % 100 == 0:
                            file_writing_obj = open(
                                './' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(
                                    configs.high) + '.txt', 'w')
                            file_writing_obj.write(str(log))

                    rewards.append(np.mean(ep_rewards).item())
                    losses.append(loss)
                    critic_loss.append(v_loss)

                    cost = env.mchsEndTimes.max(-1).max(-1)
                    costs.append(cost.mean())

                    # Only save models and print progress on rank 0
                    if rank == 0 and (batch_idx + 1) % 20 == 0:
                        end = time.time()
                        times.append(end - start)
                        start = end
                        mean_loss = np.mean(losses[-20:])
                        mean_reward = np.mean(costs[-20:])
                        critic_losss = np.mean(critic_loss[-20:])

                        print(f'Rank {rank}, Batch {batch_idx}/{len(train_loader)}, '
                              f'reward: {mean_reward:.3f}, loss: {mean_loss:.4f}, '
                              f'critic_loss: {critic_losss:.4f}, took: {times[-1]:.4f}s')
                        
                        # Save checkpoints
                        filepath = 'saved_network'
                        filename = f'FJSP_J{configs.n_j}M{configs.n_m}'
                        filepath = os.path.join(filepath, filename)
                        epoch_dir = os.path.join(filepath, f'{100}_{batch_idx}')
                        if not os.path.exists(epoch_dir):
                            os.makedirs(epoch_dir)
                        
                        # Save the unwrapped model state dict
                        torch.save(ppo.policy_job.module.state_dict(), 
                                 os.path.join(epoch_dir, 'policy_job.pth'))
                        torch.save(ppo.policy_mch.module.state_dict(),
                                 os.path.join(epoch_dir, 'policy_mch.pth'))

                        # Validation
                        validation_log = validate(valid_loader, local_batch_size, ppo.policy_job, ppo.policy_mch).mean()
                        if validation_log < record:
                            epoch_dir = os.path.join(filepath, 'best_value100')
                            if not os.path.exists(epoch_dir):
                                os.makedirs(epoch_dir)
                            torch.save(ppo.policy_job.module.state_dict(),
                                     os.path.join(epoch_dir, 'policy_job.pth'))
                            torch.save(ppo.policy_mch.module.state_dict(),
                                     os.path.join(epoch_dir, 'policy_mch.pth'))
                            record = validation_log
                            print(f'New best validation score: {validation_log}')

            if rank == 0:
                np.savetxt(f'./N_{configs.n_j}_M{configs.n_m}_u100', costs, delimiter="\n")

        # Wait for all processes to complete
        dist.barrier()
        if rank == 0:
            print(f"Training completed on all {world_size} GPUs")
            
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup()

def main(epochs):
    # Set memory management configurations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs!")
        
        # Configure PyTorch memory management
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        # Set TF32 for better memory efficiency
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Launch training processes
        try:
            mp.spawn(train,
                    args=(n_gpus,),
                    nprocs=n_gpus,
                    join=True)
        except Exception as e:
            print(f"Error in distributed training: {str(e)}")
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            raise e
    else:
        print("No GPUs available. Running on CPU.")
        train(0, 1)  # Run on CPU

if __name__ == "__main__":
    # Fix multiprocessing and memory issues
    torch.multiprocessing.set_start_method('spawn')
    
    # Set environment variables for better memory management
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        main(1)
    except Exception as e:
        # Ensure proper cleanup
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        raise e
