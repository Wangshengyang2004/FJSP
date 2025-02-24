import torch
import traceback

def get_best_device():
    """Auto-detect the best available device between CUDA, MPS, and CPU."""
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA device.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS (Apple Silicon GPU) is available. Testing sparse tensor operations...")
        try:
            # Test if sparse operations work on MPS
            device = torch.device("mps")
            idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
            val = torch.tensor([1., 1.], device=device)
            test_sparse = torch.sparse_coo_tensor(idx.t(), val, (2, 2), device=device)
            # If we get here, sparse operations work
            print("Sparse tensor operations work on MPS. Using MPS device.")
            return device
        except Exception as e:
            print("\nFalling back to CPU due to MPS limitations:")
            print(f"Error details: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
            print("\nThis is expected as MPS backend currently doesn't fully support sparse tensor operations.")
            return torch.device("cpu")
    
    print("Neither CUDA nor MPS available. Using CPU device.")
    return torch.device("cpu") 