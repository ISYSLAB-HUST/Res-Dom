import numpy as np
import os

from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

seq_length = 700  


def sensitivity(y_true, y_pred):
    true_label = K.argmax(y_true, axis=-1)
    pred_label = K.argmax(y_pred, axis=-1)
    INTERESTING_CLASS_ID = 2
    sample_mask = K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')

    TP_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
    TP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask    
    TP = K.sum(TP_tmp1 * TP_tmp2)

    FN_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
    FN_tmp2 = K.cast(K.not_equal(pred_label, 0), 'int32') * sample_mask    
    FN = K.sum(FN_tmp1 * FN_tmp2)

    epsilon = 0.000000001
    return K.cast(TP, 'float') / (K.cast(TP, 'float') + K.cast(FN, 'float') + epsilon)


def precision(y_true, y_pred):
    true_label = K.argmax(y_true, axis=-1)
    pred_label = K.argmax(y_pred, axis=-1)
    INTERESTING_CLASS_ID = 2
    sample_mask = K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')

    TP_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
    TP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask
    TP = K.sum(TP_tmp1 * TP_tmp2)

    FP_tmp1 = K.cast(K.not_equal(true_label, 0), 'int32') * sample_mask
    FP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask
    FP = K.sum(FP_tmp1 * FP_tmp2)

    epsilon = 0.000000001
    return K.cast(TP, 'float') / (K.cast(TP, 'float') + K.cast(FP, 'float') + epsilon)


def f1_score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    sen = sensitivity(y_true, y_pred)
    epsilon = 0.000000001
    f1 = 2 * pre * sen / (pre + sen + epsilon)
    return f1


def read_hmm(chain, data_dir):
    from Bio import SeqIO
    import pandas as pd
    fasta_file = os.path.join(data_dir, chain, "seq.fasta")
    fname = os.path.join(data_dir, chain, chain+".ohmm")
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


def load_ss(chain, data_path):
    """
    func: load ss fea array from ss file
    """
    ss_vector = {"C": [1, 0, 0], "H": [0, 1, 0], "E": [0, 0, 1]}
    ss_f = os.path.join(data_path, chain, chain+".ss")
    ss_fea_arr = []
    f = open(ss_f, "r")
    ss_str = f.readlines()[-1].strip()
    f.close()
    for i in ss_str:
        ss_fea_arr.append(ss_vector[i])
    return np.array(ss_fea_arr)


def load_rsa(chain, data_path):
    """
    func: load rsa fea array from acc file
    """
    rsa_vector = {"e": [1, 0], "-": [0, 1], "b": [0, 1]}
    rsa_f = os.path.join(data_path, chain, chain+".acc")
    rsa_fea_arr = []
    f = open(rsa_f, "r")
    rsa_str = f.readlines()[-1].strip()
    f.close()
    for i in rsa_str:
        rsa_fea_arr.append(rsa_vector[i])
    return np.array(rsa_fea_arr)


def data(chain, data_path):
    """
    param: casp_ls -- seq list of casp12 set 
    func: concatenate 1280d esm feature, 30d hmm and 5d ss_rsa
    return: 
    """
    import torch
    from keras.preprocessing.sequence import pad_sequences
    MAX_LEN = 700 
    EMB_LAYER = 34
    x1_test = []
    y_test = []
    hmm_mean = np.load(os.path.join(data_path, "npy_data", "hmm_mean.npy"))
    hmm_std = np.load(os.path.join(data_path, "npy_data", "hmm_std.npy"))
    # 1280+30+5 esm+hmm+ss+rsa
    hmm_arr = read_hmm(chain, data_path)
    # shape: (len, 1280)
    fn = os.path.join(data_path, chain, chain+".pt")
    embs = torch.load(fn)
    esm_feature_arr = np.array(embs['representations'][EMB_LAYER])
    # normlization
    hmm_norm_arr = (hmm_arr-hmm_mean)/hmm_std
    ss_arr = load_ss(chain, data_path)
    rsa_arr = load_rsa(chain, data_path)
    # 1280+30+5
    combine_feature_arr = np.concatenate((esm_feature_arr, hmm_norm_arr, ss_arr, rsa_arr), axis=1)
    combine_feature_arr = combine_feature_arr.reshape(1, combine_feature_arr.shape[0], combine_feature_arr.shape[1])
    combine_feature_arr = pad_sequences(combine_feature_arr, MAX_LEN, dtype="float32", padding='post', truncating='post')
    x1_test = combine_feature_arr
    x1_test = np.concatenate((x1_test, combine_feature_arr), axis=0)
    return x1_test

from keras.models import Model, load_model
from keras.layers import Input, Conv1D, BatchNormalization
from keras.layers import Activation, Dense, Dropout
from keras.layers import Dropout, GRU, LSTM, TimeDistributed
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import concatenate, add
from keras import regularizers


def predict_dom(score, threshold=0.5, min_len=10):
    merge_dom_len = 40
    # find under_threshold continuous region
    under_threshold_ls = np.where(score < threshold)[0]
    if under_threshold_ls == []:
        return []
    block_ls = to_block(under_threshold_ls[:])
    block_filter_ls = [x for x in block_ls if len(x)>min_len]
    # merge region
    predict_boundary_ls = [x[len(x)//2] for x in block_filter_ls]
    if len(predict_boundary_ls) == 1:
        return predict_boundary_ls
    del_bound_ls = []
    ind = 0
    while ind <= len(predict_boundary_ls)-2:
        if predict_boundary_ls[ind+1] - predict_boundary_ls[ind] < 40:
            del_bound_ls.append((ind, ind+1))
            ind += 2
        else:
            ind += 1
    predict_boundary_ls_copy = predict_boundary_ls[:]
    for pair_set in del_bound_ls:
        a = block_filter_ls[pair_set[0]]
        b = block_filter_ls[pair_set[1]]
        val_a = min([score[i] for i in a])
        val_b = min([score[i] for i in b])
        if val_a >= val_b:
            predict_boundary_ls.remove(predict_boundary_ls_copy[pair_set[0]])
        else:
            predict_boundary_ls.remove(predict_boundary_ls_copy[pair_set[1]])
    return predict_boundary_ls



def to_block(input_ls):
    """
    func: 连续数字区域
    """
    output_ls = []
    while input_ls != []:
        for i in range(len(input_ls)-1, -1, -1):
            if (input_ls[i] - input_ls[0]) == i:
                output_ls.append(input_ls[:i+1].tolist())
                input_ls = input_ls[i+1:]
                break
    return output_ls


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    # load data
    chain = "T0987"
    x_test = data(chain, file_path)
    # load model
    model = load_model("res-dom.h5", custom_objects={'precision': precision, 'sensitivity': sensitivity, 'f1_score':f1_score})
    # model =multi_gpu_model(model, gpus=2)    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001),metrics=['accuracy',precision, sensitivity, f1_score])
    preds=model.predict(x_test)
    # The parallel mode was used when training the model.
    chain_ls = [chain, chain]
    for ind, chain in enumerate(chain_ls):
        scorepath = os.path.join(file_path, chain, chain+'.score')
        score = preds[ind,:,:]
        from Bio import SeqIO
        fasta_f = os.path.join(file_path , chain, "seq.fasta")
        seq_record = SeqIO.parse(fasta_f, "fasta")
        l = len(next(seq_record).seq)
        if l > 700:
            l=700
        score = score[:,1]
        score = score[:l]
        np.savetxt(scorepath, score)
        predict_boundary_ls = predict_dom(score, 0.5, 20)
        # return predicted result
        dom_num = len(predict_boundary_ls)+1
        dom_ls = []
        dom_str = ""
        start = 1
        if dom_num == 1:
            dom_str = "({}-{})".format(start, l)
        else:
            for ind, bound in enumerate(predict_boundary_ls):
                single_dom = "({}-{})".format(start, bound)
                dom_ls.append(single_dom)
                start = bound+1
            single_dom = "({}-{})".format(start, l)
            dom_ls.append(single_dom)
            dom_str = "".join(dom_ls)
    print(chain, l, dom_num, dom_str, sep="\t", end="\n")


