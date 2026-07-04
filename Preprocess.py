import os
import torch
import numpy as np
import pickle
from Params import args
from VirtualInteraction import compute_multimodal_similarity, find_topk_similar, create_virtual_interactions, save_augmented_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadFeatures(filename):
    feats = np.load(filename)
    return torch.tensor(feats).float().to(device), np.shape(feats)[1]

def loadOneFile(filename):
    import scipy.sparse as sp
    with open(filename, 'rb') as fs:
        ret = (pickle.load(fs) != 0).astype(np.float32)
    if type(ret) != sp.coo_matrix:
        ret = sp.coo_matrix(ret)
    return ret

def run_preprocess(dataset_path):
    predir = os.path.join(dataset_path, args.data) + '/'
    trnfile = predir + 'trnMat.pkl'
    
    imagefile = predir + 'image_feat.npy'
    textfile = predir + 'text_feat.npy'
    audiofile = predir + 'audio_feat.npy' if args.data == 'tiktok' else None
    
    print(f"Loading data for {args.data}...")
    trnMat = loadOneFile(trnfile)
    
    image_feats, _ = loadFeatures(imagefile)
    text_feats, _ = loadFeatures(textfile)
    audio_feats = None
    if audiofile:
        audio_feats, _ = loadFeatures(audiofile)
        
    print("Computing multimodal similarity matrix...")
    sim_matrix = compute_multimodal_similarity(image_feats, text_feats, audio_feats, alpha=args.vi_alpha)
    
    print(f"Finding top {args.vi_topk} similar items...")
    topk_items = find_topk_similar(sim_matrix, k=args.vi_topk)
    
    print("Creating virtual interactions...")
    aug_mat = create_virtual_interactions(trnMat, topk_items, args.vi_lambda, w_s=1.0)
    
    out_file = os.path.join(predir, f'augmented_adj_matrix_{args.data}.pkl')
    save_augmented_matrix(aug_mat, out_file)
    print(f"Preprocessing completed for {args.data}")

if __name__ == '__main__':
    dataset_path = getattr(args, 'dataset_path', './Datasets')
    run_preprocess(dataset_path)
