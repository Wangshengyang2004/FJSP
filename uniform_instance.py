import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel

def permute_rows(x):
    '''
    Randomly permutes elements in each row of a 2D numpy array
    
    Args:
        x: 2D numpy array to be permuted
        
    Returns:
        Permuted array where each row has its elements randomly reordered
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high, seed=None):
    '''
    Generates a single FJSP instance with random processing times
    
    Args:
        n_j: Number of jobs
        n_m: Number of machines 
        low: Lower bound for processing times
        high: Upper bound for processing times
        seed: Random seed for reproducibility
        
    Returns:
        3D numpy array of shape (n_j, n_m, n_m) containing processing times.
        times[j,o,m] represents processing time of operation o of job j on machine m.
        A value of 0 indicates machine m cannot process that operation.
    '''
    if seed != None:
        np.random.seed(seed)

    time0 = np.random.randint(low=low, high=high, size=(n_j, n_m,n_m-1))
    time1 = np.random.randint(low=1, high=high, size=(n_j, n_m,1))
    times = np.concatenate((time0,time1),-1)
    for i in range(n_j):
        times[i] = permute_rows(times[i])
    return times

#-99~99 randint -- 5
#-99~99 0~99 uniform -- 0
#-99~99 1~99 uniform -- 0


class FJSPDataset(Dataset):
    '''
    PyTorch Dataset class for generating multiple FJSP instances
    
    Args:
        n_j: Number of jobs
        n_m: Number of machines
        low: Lower bound for processing times 
        high: Upper bound for processing times
        num_samples: Number of instances to generate
        seed: Random seed for reproducibility
        offset: Offset for indexing (not used)
        distribution: Distribution type (not used)
    '''

    def __init__(self, n_j, n_m, low, high, num_samples=1000000, seed=None, offset=0, distribution=None):
        super(FJSPDataset, self).__init__()

        self.data_set = []
        if seed != None:
            np.random.seed(seed)
            
        # Generate random processing times
        time0 = np.random.uniform(low=low, high=high, size=(num_samples, n_j, n_m, n_m - 1))
        time1 = np.random.uniform(low=0, high=high, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        
        # Randomly permute machine assignments for each operation
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = permute_rows(times[j][i])
                
        self.data = np.array(times)
        self.size = len(self.data)

    def getdata(self):
        '''Returns the full dataset'''
        return self.data

    def __len__(self):
        '''Returns number of instances in dataset'''
        return self.size

    def __getitem__(self, idx):
        '''Returns instance at given index'''
        return self.data[idx]

'''a = TSPDataset(3,3,-1,1,200,2000)
a = a.getdata()
print(a)
for t in range(2000):
    for i in range(3):
        for j in range(3):
            durmch = a[t][i][j][torch.where(a[t][i][j] > 0)]
            a[t][i][j] = torch.tensor([durmch.mean() if i < 0 else i for i in a[t][i][j]])

print(a)'''
def override(fn):
    """
    override decorator
    """
    return fn

if __name__ == '__main__':
    '''
    Test script to demonstrate dataset generation and visualization
    
    Generates a small FJSP dataset, saves it to file, loads it back,
    and prints detailed information about the instances including:
    - Processing times
    - Available machines for each operation
    - Basic statistics
    '''
    # Generate dataset
    n_j = 3  # 3 jobs
    n_m = 3  # 3 machines
    low = -1
    high = 1
    seed = 42
    num_samples = 5  # Generate 5 instances for demonstration
    
    print("\n1. Generating and saving dataset:")
    dataset = FJSPDataset(n_j, n_m, low, high, num_samples, seed)
    data = dataset.data
    
    # Save the dataset
    filename = f"FJSP_J{n_j}M{n_m}_test_data.npy"
    np.save(filename, data)
    print(f"Dataset saved to {filename}")
    
    # Load and verify the dataset
    print("\n2. Loading and verifying saved dataset:")
    loaded_data = np.load(filename)
    print(f"Shape of loaded data: {loaded_data.shape}")
    print(f"Data type: {loaded_data.dtype}")
    
    # Display the data in a structured format
    print("\n3. Displaying loaded data:")
    for instance_idx in range(len(loaded_data)):
        print(f"\nInstance {instance_idx + 1}:")
        for job_idx in range(n_j):
            print(f"\nJob {job_idx + 1}:")
            for op_idx in range(n_m):
                processing_times = loaded_data[instance_idx, job_idx, op_idx]
                available_machines = np.where(processing_times > 0)[0]
                print(f"  Operation {op_idx + 1}:")
                print(f"    Processing times: {processing_times}")
                print(f"    Available machines: {available_machines + 1}")
                print(f"    Number of available machines: {len(available_machines)}")
    
    # Verify data consistency
    print("\n4. Data consistency check:")
    if np.array_equal(data, loaded_data):
        print("✓ Saved and loaded data are identical")
    else:
        print("✗ Warning: Saved and loaded data differ")
    
    # Basic statistics
    print("\n5. Data statistics:")
    print(f"Min processing time: {loaded_data[loaded_data > 0].min():.2f}")
    print(f"Max processing time: {loaded_data.max():.2f}")
    print(f"Mean processing time (excluding zeros): {loaded_data[loaded_data > 0].mean():.2f}")
    print(f"Total number of operations: {n_j * n_m * num_samples}")
