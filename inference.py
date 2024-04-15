import torch.nn.functional
from typing import List, Union
import numpy as np
import torch
from DeepFMPO.build_vocab import hybrid_encode_molecule, smiles_encode_molecule
from dataset import load_encoders
from model import PredictionModel
from torch.nn.utils.rnn import pad_sequence
from device import device


class InferenceModel:
    def __init__(self, finetuned_weights_path: str) -> None:
        finetuned_weights_file = torch.load(finetuned_weights_path, map_location='cpu')

        arch = finetuned_weights_file['model_arch']
        num_layers = arch['num_layers']
        num_heads = arch['num_heads']
        d_model = arch['d_model']
        dff = d_model * 4

        self.fragmentation = finetuned_weights_file['fragmentation']
        smiles_vocab_path = finetuned_weights_file['smiles_vocab_path']
        smiles_frag_vocab_path = finetuned_weights_file['smiles_frag_vocab_path']
        frag_vocab_path = finetuned_weights_file['frag_vocab_path']

        self.reg_cols = finetuned_weights_file['reg_cols']
        self.clf_cols = finetuned_weights_file['clf_cols']
        self.reg_means = finetuned_weights_file['reg_means']
        self.reg_stds = finetuned_weights_file['reg_stds']

        self.model_str2num, self.smiles_str2num, self.frag_str2num = load_encoders(self.fragmentation,
                                                                smiles_vocab_path=smiles_vocab_path,
                                                                smiles_frag_vocab_path=smiles_frag_vocab_path,
                                                                frag_vocab_path=frag_vocab_path)

        if self.fragmentation:
            vocab_size = len(self.model_str2num) + len(self.smiles_str2num) + len(self.frag_str2num)
        else:
            vocab_size = len(self.model_str2num) + len(self.smiles_str2num)

        self.model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff,
                                     num_heads=num_heads, vocab_size=vocab_size,
                                     dropout_rate=0.0, reg_nums=len(self.reg_cols),
                                     clf_nums=len(self.clf_cols), maximum_positional_encoding=200)
        self.model.load_state_dict(finetuned_weights_file['model_state_dict'])
        self.model = self.model.to(device)
        self.model = self.model.eval()
    
    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding
    
    def __call__(self, seqs: Union[str, List[str]], format_output: bool = True, clf_threshold: float = 0.5):
        """
        Arguments:
            seqs: SMILES sequences
            format_output: receive raw model output (False) or formatted and re-scaled (True)
            clf_threshold: threshold for true/false classification after applying logistic function
        """
        if isinstance(seqs, str):
            seqs = [seqs]
        
        nums_list = []
        for seq in seqs:
            seq_encoding = self._char_to_idx(seq=seq)
            total_pred_cols = len(self.reg_cols) + len(self.clf_cols)
            if total_pred_cols > 0:
                ps = [f'<p{i}>' for i in range(total_pred_cols)]
                seq_encoding = [self.model_str2num[p] for p in ps] + seq_encoding
            nums_list.append(np.array(seq_encoding).astype('int32'))
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in nums_list], batch_first=True).long()
        xs = xs.to(device)

        with torch.no_grad():
            output = self.model(xs)

        if format_output:
            res = {}
            # Setting up results dictionary with clf and reg task names
            for i in range(len(output['clf'][0])):
                res[self.clf_cols[i]] = []
            
            for i in range(len(output['reg'][0])):
                res[self.reg_cols[i]] = []
            
            for clf_out in output['clf']:
                for i in range(len(clf_out)):
                    res[self.clf_cols[i]].append(torch.sigmoid(clf_out[i]).item() > clf_threshold)
            
            for reg_out in output['reg']:
                for i in range(len(reg_out)):
                    res[self.reg_cols[i]].append((reg_out[i].item() * self.reg_stds[i]) + self.reg_means[i])
            return res
        
        return output
        

if __name__ == "__main__":
    # NOTE: device.py needs to be changed, depending on where inference is performed (cpu/cuda)
    finetuned_model_path = 'weights/finetune/15-04-2024-14-58-51_1000freq_onephase_best.pt'
    model = InferenceModel(finetuned_model_path)

    example_smis = ["CCC", "CCCCCC"]
    example_smi = "CCC"
    
    print(model(example_smis))
    print("\n############\n")
    print(model(example_smi))