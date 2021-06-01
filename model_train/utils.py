from Bio import SeqIO
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def read_pssm(chain):
    fasta_dir = "/home/haolinz/workspace/esm-dnndom/workspace/makeLabel/fasta_dir"
    pssm_dir = "/home/haolinz/workspace/esm-dnndom/workspace/makeLabel/hmm_dir"
    fasta_file = os.path.join(fasta_dir, chain+".fasta")
    fname = os.path.join(pssm_dir, chain, chain+".pssm")
    seq_record = SeqIO.parse(fasta_file, "fasta")
    seq = next(seq_record).seq
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname, 'r') as f:
        tmp_pssm = pd.read_csv(f, delim_whitespace=True,
                               names=pssm_col_names).dropna().values[:, 2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm


def load_ss_rsa_fea(chain):
    # 读取后五维fea
    # /home/haolinz/workspace/esm-dnndom/workspace/train_fea/2o6pA/2o6pA.fea 
    """
    param: chain  -- name of pdb chain
    func: read last 5 dimension vector from chain.fea    (secondary structure and relative solvent accessibility )
    return: 2d array -- shape (len, 5)
    """
    fea_dir = "/home/haolinz/workspace/esm-dnndom/workspace/train_fea"
    fea_file = os.path.join(fea_dir, chain, chain+".fea")
    fea_arr = np.loadtxt(fea_file)
    ss_rsa_fea = fea_arr[:, 20:]
    return ss_rsa_fea


def read_hmm(chain):
    fasta_dir = "/home/haolinz/workspace/esm-dnndom/workspace/makeLabel/fasta_dir"
    pssm_dir = "/home/haolinz/workspace/esm-dnndom/workspace/makeLabel/hmm_dir"
    fasta_file = os.path.join(fasta_dir, chain+".fasta")
    fname = os.path.join(pssm_dir, chain, chain+".ohmm")
    seq_record = SeqIO.parse(fasta_file, "fasta")
    seq = next(seq_record).seq
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname, 'r') as f:
        hhm = pd.read_csv(f, delim_whitespace=True, names=hhm_col_names)
    pos1 = (hhm['0'] == 'HMM').idxmax() + 3
    hhm = hhm[pos1:-1].values[:, :num_hhm_cols].reshape([-1, 44])
    hhm[hhm == '*'] = '9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:, 2:-12].astype(float)