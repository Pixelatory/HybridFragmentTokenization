import math
import pickle
from typing import Iterator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from DeepFMPO.build_vocab import combine_vocabs, load_vocab_from_file, get_hybrid_encoders, \
    hybrid_encode_molecule, smiles_encode_molecule

def load_encoders(fragmentation, smiles_vocab_path, smiles_frag_vocab_path, frag_vocab_path, num_task_specific_tokens = 100):
    if fragmentation:
        smiles_vocab = combine_vocabs(
            load_vocab_from_file(smiles_vocab_path) + load_vocab_from_file(smiles_frag_vocab_path))
    else:
        smiles_vocab = load_vocab_from_file(smiles_vocab_path)

    if fragmentation:
        frag_vocab = load_vocab_from_file(frag_vocab_path)
    else:
        frag_vocab = None

    # add task-specific tokens for MTL-BERT
    additional_model_tokens = []
    for i in range(num_task_specific_tokens):
        additional_model_tokens.append(f'<p{i}>')

    # if fragmentation is false, just don't use frag_str2num and you're good to go
    return get_hybrid_encoders(smiles_vocab, frag_vocab, additional_model_tokens)


def randomize_smile(sml):
    m = Chem.MolFromSmiles(sml)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles


def canonical_smile(sml):
    m = Chem.MolFromSmiles(sml)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles


class Smiles_Bert_Dataset(Dataset):
    def __init__(self, path, smiles_col, model_str2num, smiles_str2num, frag_str2num, fragmentation):
        if path.endswith('txt'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)

        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num
        self.frag_str2num = frag_str2num

        self.data = self.df[smiles_col].to_numpy().reshape(-1).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights

    def numerical_smiles(self, smiles):
        nums_list = self._char_to_idx(smiles)
        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = self.model_str2num['<MASK>']
            elif rand < 0.9:
                if self.fragmentation:
                    nums_list[i] = int(
                        np.random.randint(len(self.model_str2num), len(self.smiles_str2num) + len(self.frag_str2num)))
                else:
                    nums_list[i] = int(np.random.randint(len(self.model_str2num), len(self.smiles_str2num)))
        x = np.array(nums_list).astype('int64')
        weights = weight.astype('float32')
        return x, y, weights

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding

    def pickle_data_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.data, f)


class Prediction_Dataset(object):
    def __init__(self, df, fragmentation, model_str2num, smiles_str2num, frag_str2num, smiles_head='SMILES',
                 reg_heads=None, clf_heads=None, boundaries=None):
        if clf_heads is None:
            clf_heads = []
        if reg_heads is None:
            reg_heads = []

        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads

        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()

        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')
        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num
        self.frag_str2num = frag_str2num
        self.boundaries = boundaries

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]

        properties = [None, None]
        if len(self.clf_heads) > 0:
            clf = self.clf[item]
            properties[0] = clf

        if len(self.reg_heads) > 0:
            reg = self.reg[item]
            properties[1] = reg

        nums_list = self._char_to_idx(seq=smiles)
        if len(self.reg_heads) + len(self.clf_heads) > 0:
            ps = [f'<p{i}>' for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [self.model_str2num[p] for p in ps] + nums_list
        x = np.array(nums_list).astype('int32')
        return x, properties

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding
    
    def get_boundaries(self):
        return self.boundaries


class Pretrain_Collater():
    def __init__(self):
        super(Pretrain_Collater, self).__init__()

    def __call__(self, data):
        xs, ys, weights = zip(*data)

        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long()
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long()
        weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights], batch_first=True).float()

        return xs, ys, weights


class Finetune_Collater():
    def __init__(self, clf_cols, reg_cols):
        super(Finetune_Collater, self).__init__()
        self.clf_cols = clf_cols
        self.reg_cols = reg_cols

    def __call__(self, data):
        xs, properties_list = zip(*data)
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long()
        properties_dict = {'clf': None, 'reg': None}

        if len(self.clf_cols) > 0:
            properties_dict['clf'] = torch.from_numpy(
                np.concatenate([p[0].reshape(1, -1) for p in properties_list], 0).astype('int32'))

        if len(self.reg_cols) > 0:
            properties_dict['reg'] = torch.from_numpy(
                np.concatenate([p[1].reshape(1, -1) for p in properties_list], 0).astype('float32'))

        return xs, properties_dict


def kfolds(datas, k: int):
    """
        Generates k similarly sized folds (may not be equally sized)
        Note: doesn't shuffle, so do the shuffle before calling this
    """
    result = []
    for data in datas:
        fold_size = math.ceil(len(data) / k)
        tmp_folds = []
        for i in range(k):
            tmp_folds.append(data[fold_size * i: fold_size * (i + 1)].reset_index(drop=True))
        result.append(tmp_folds)
    return result

def train_test_splits(datas, train_ratio: float):
    result = []
    for data in datas:
        train, test = train_test_split(data, train_size=train_ratio, shuffle=False)
        result.append((train, test))
    return result

class DisjointDatasetBatchSampler(Sampler[int]):
    """
        Stratified sampling from disjoint datasets
        To best of abilities, try to make each batch contain at least 1 sample from each dataset.
    """
    def __init__(self, dataset_boundaries: List[Tuple[int, int]], 
                 batch_size: int, shuffle: bool, generator: np.random.Generator=None) -> None:
        super().__init__(dataset_boundaries)
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle = shuffle
        self.dataset_boundaries = dataset_boundaries

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer "
                             "value, but got batch_size={}".format(self.batch_size))
        if self.batch_size < len(self.dataset_boundaries):
            raise ValueError("batch_size should be a value >= than the number of disjoint datasets")

    def __len__(self):
        """
        Returns: Number of batches
        """
        return math.ceil(self.dataset_boundaries[-1][1] / self.batch_size)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            generator = np.random.default_rng()
        else:
            generator = self.generator

        num_dataset_samples = self.dataset_boundaries[-1][1]
        num_of_batches = math.ceil(num_dataset_samples / self.batch_size)
        batches = [[] for _ in range(num_of_batches)]

        dataset_idxs = []

        for dataset_boundary in self.dataset_boundaries:
            diff = dataset_boundary[1] - dataset_boundary[0] + 1
            if self.shuffle:
                dataset_idxs.append((generator.permutation(diff) + dataset_boundary[0]).tolist())
            else:
                dataset_idxs.append([i + dataset_boundary[0] for i in range(diff)])
        
        for idxs in dataset_idxs:
            cursor = 0
            for i in range(len(idxs)):
                while True:
                    curr_batch = cursor % len(batches)
                    cursor += 1
                    if len(batches[curr_batch]) < self.batch_size:
                        batches[curr_batch].append(idxs[i])
                        break
        
        if self.shuffle:
            for batch in batches:
                batch = generator.permutation(batch).tolist()

        for batch in batches:
            yield batch