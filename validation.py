from epsGreedyForMch import PredictMch
from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSP_Env import FJSP,DFJSP_GANTT_CHART
from mb_agg import g_pool_cal
import copy
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.device_utils import get_best_device
import os

# Set matplotlib to use a backend that doesn't require a display
import matplotlib
matplotlib.use('Agg')

# Configure default font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Get device once at module level
DEVICE = get_best_device()

def validate(vali_set,batch_size, policy_jo,policy_mc):
    policy_job = copy.deepcopy(policy_jo)
    policy_mch = copy.deepcopy(policy_mc)
    policy_job.eval()
    policy_mch.eval()
    
    # Create directory for Gantt charts if it doesn't exist
    gantt_dir = 'gantt_charts'
    if not os.path.exists(gantt_dir):
        os.makedirs(gantt_dir)
        
    def eval_model_bat(bat,i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=configs.n_j, n_m=configs.n_m)
            gantt_chart = DFJSP_GANTT_CHART(configs.n_j, configs.n_m)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=DEVICE)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(DEVICE)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(DEVICE)
            pool=None
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(DEVICE).to_sparse(), configs.n_j * configs.n_m)
                env_fea = torch.from_numpy(np.copy(fea)).float().to(DEVICE)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(DEVICE)
                env_mask = torch.from_numpy(np.copy(mask)).to(DEVICE)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(DEVICE)
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea,
                                                                                               graph_pool=g_pool_step,
                                                                                               padded_nei=None,
                                                                                               adj=env_adj,
                                                                                               candidate=env_candidate
                                                                                               , mask=env_mask
                                                                                               , mask_mch=env_mask_mch
                                                                                               , dur=env_dur
                                                                                               , a_index=0
                                                                                               , old_action=0
                                                                                                ,mch_pool=pool
                                                                                               ,old_policy=True,
                                                                                                T=1
                                                                                               ,greedy=True
                                                                                               )

                pi_mch,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a,gantt_chart)

                j += 1
                if env.done():
                    # Save the Gantt chart with proper styling
                    plt.title(f'Instance {i+1} - Makespan: {env.mchsEndTimes.max(-1).max(-1)[0]:.2f}', pad=20)
                    # Add grid for both x and y axis
                    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
                    # Ensure y-axis shows integer ticks only
                    ax = plt.gca()
                    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                    plt.tight_layout()
                    plt.savefig(os.path.join(gantt_dir, f'gantt_chart_instance_{i+1}.png'), 
                              format='png', 
                              dpi=150, 
                              bbox_inches='tight',
                              facecolor='white',
                              edgecolor='none')
                    plt.close()  # Close the figure to free memory
                    break
            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)

    totall_cost = torch.cat([eval_model_bat(bat,i) for i,bat in enumerate(vali_set)], 0)
    return totall_cost



if __name__ == '__main__':

    from uniform_instance import uni_instance_gen,FJSPDataset
    import numpy as np
    import time
    import argparse
    from Params import configs

    Pn_j = 5  # Number of jobs of instances to test
    Pn_m = 3  # Number of machines instances to test
    Nn_j = 3  # Number of jobs on which to be loaded net are trained
    Nn_m = 3  # Number of machines on which to be loaded net are trained
    low = -99  # LB of duration
    high = 99  # UB of duration
    seed = 200  # Cap seed for validate set generation
    n_vali = 100  # Validation set size
    load_data = True  # Load validation data from file instead of generating new data
    data_file = "FJSP_J3M3_test_data.npy"  # Path to the validation data file (if loading from file)

    N_JOBS_P = Pn_j
    N_MACHINES_P = Pn_m
    LOW = low
    HIGH = high
    N_JOBS_N = Nn_j
    N_MACHINES_N = Nn_m
    from torch.utils.data import DataLoader
    from PPOwithValue import PPO
    import torch
    import os
    from torch.utils.data import Dataset
    
    print(f"\nUsing device: {DEVICE}")
    
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    filepath = 'saved_network'
    filepath = os.path.join(filepath, 'FJSP_J%sM%s' % (3,configs.n_m))
    filepath = os.path.join(filepath, 'best_value0')

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')

    job_path = os.path.join(filepath,job_path)
    mch_path = os.path.join(filepath, mch_path)

    # Load state dicts with weights_only=True for security
    ppo.policy_job.load_state_dict(torch.load(job_path, weights_only=True))
    ppo.policy_mch.load_state_dict(torch.load(mch_path, weights_only=True))
    num_val = 10
    batch_size = 1
    SEEDs = [200]
    result = []

    for SEED in SEEDs:
        mean_makespan = []
        if configs.load_data:
            if configs.data_file is None:
                # Use default filename if not specified
                data_file = f"FJSP_J{configs.n_j}M{configs.n_m}_test_data.npy"
            else:
                data_file = configs.data_file
                
            print(f"Loading validation data from: {data_file}")
            try:
                validat_dataset = np.load(file=data_file)
                print(f"Loaded dataset with shape: {validat_dataset.shape}")
            except FileNotFoundError:
                print(f"Error: Could not find data file {data_file}")
                print("Falling back to generating new validation dataset...")
                validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, num_val, SEED)
        else:
            print("Generating new validation dataset...")
            validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, num_val, SEED)
        
        valid_loader = DataLoader(validat_dataset, batch_size=batch_size)
        vali_result = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch)
        
        print("\nValidation Results:")
        print("Individual instance makespans:")
        for i, makespan in enumerate(vali_result, 1):
            print(f"Instance {i}: {makespan:.2f}")
        print(f"\nAverage makespan: {np.array(vali_result).mean():.2f}")
        print(f"Best makespan: {np.array(vali_result).min():.2f}")
        print(f"Worst makespan: {np.array(vali_result).max():.2f}")

    # print(min(result))

