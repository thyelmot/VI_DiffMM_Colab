import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_similarity(feat1, feat2):
    """
    Computes cosine similarity between two feature matrices using batching if needed
    """
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    return torch.mm(feat1, feat2.t())

def compute_multimodal_similarity(image_feats, text_feats, audio_feats=None, alpha=0.5):
    """
    Computes multimodal similarity matrix for items.
    """
    sim_v = compute_similarity(image_feats, image_feats)
    sim_t = compute_similarity(text_feats, text_feats)
    
    if audio_feats is not None:
        sim_a = compute_similarity(audio_feats, audio_feats)
        return (sim_v + sim_t + sim_a) / 3.0
    
    return alpha * sim_v + (1 - alpha) * sim_t

def find_topk_similar(sim_matrix, k):
    """
    Find top-k similar items for each item based on similarity matrix.
    Returns indices tensor of shape (num_items, k)
    """
    # Set self-similarity to -1 to avoid picking the item itself
    sim_matrix.fill_diagonal_(-1)
    topk_vals, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
    return topk_indices

def create_virtual_interactions(trnMat, topk_items, lambda_vi, w_s=1.0):
    """
    Synergistic Strategy: Create virtual interactions by adding edges to top-k similar items
    of items the user already interacted with.
    """
    # trnMat is (num_users, num_items) sparse matrix
    num_users, num_items = trnMat.shape
    
    # We will compute the new interactions. Since trnMat is sparse, we can iterate over users.
    rows, cols = trnMat.nonzero()
    
    new_rows = []
    new_cols = []
    new_data = []
    
    print("Generating virtual edges...")
    
    # Convert topk_items to CPU numpy array for faster indexing
    topk_items_np = topk_items.cpu().numpy()
    
    for r, c in tqdm(zip(rows, cols), total=len(rows)):
        # Original edge
        new_rows.append(r)
        new_cols.append(c)
        new_data.append(1.0)
        
        # Virtual edges based on item c's neighbors
        similar_items = topk_items_np[c]
        for v_item in similar_items:
            new_rows.append(r)
            new_cols.append(v_item)
            new_data.append(lambda_vi * w_s)
            
    # Create the augmented matrix
    aug_mat = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(num_users, num_items))
    
    # Sum duplicate edges (user might have multiple original items pointing to same virtual item)
    aug_mat = aug_mat.tocsr()
    
    # Confine values to max 1.0 to avoid exploding weights, but typically we just sum them.
    # The paper mentions 'Confine', which can be implemented by clipping to 1.0.
    aug_mat.data = np.clip(aug_mat.data, 0.0, 1.0)
    
    return aug_mat.tocoo()

def save_augmented_matrix(aug_mat, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(aug_mat, f)
    print(f"Saved augmented matrix to {path}")
