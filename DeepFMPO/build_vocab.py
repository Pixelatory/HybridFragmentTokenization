import re
import pandas as pd
import rdkit.Chem.Draw

from rdkit import Chem
from DeepFMPO.mol_utils import get_fragments, split_molecule


SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def build_smiles_vocab(file_name, smiles_col):
    df = pd.read_csv(file_name)
    smiles_vocab = set()
    for smi in df[smiles_col]:
        splits = re.findall(SMI_REGEX_PATTERN, smi)
        for token in splits:
            smiles_vocab.add(token)

    return list(smiles_vocab)


def build_smiles_vocab_from_fragments(file_name, smiles_col):
    """
        Fragments include additional SMILES token(s) that might
        not be seen when using build_smiles_vocab(), since some
        tokens are used to denote a linking point between fragments.
    """
    df = pd.read_csv(file_name)
    mols = []
    for smi in df[smiles_col]:
        try:
            mols.append(Chem.MolFromSmiles(smi))
        except:
            continue

    all_frags = list(get_fragments(mols)[0].keys())

    smiles_vocab = set()
    for frag in all_frags:
        for token in re.findall(SMI_REGEX_PATTERN, frag):
            smiles_vocab.add(token)

    return list(smiles_vocab)


def build_fragment_vocab(file_name, smiles_col):
    df = pd.read_csv(file_name)
    mols = []
    for smi in df[smiles_col]:
        try:
            mols.append(Chem.MolFromSmiles(smi))
        except:
            continue

    frag_vocab = list(get_fragments(mols)[0].keys())
    return frag_vocab


def get_fragment_frequencies(frag_vocab, file_name, smiles_col):
    df = pd.read_csv(file_name)

    frag_freq = {}
    for frag in frag_vocab:
        frag_freq[frag] = 0

    for entry in df[smiles_col]:
        try:
            frags = [Chem.MolToSmiles(entry) for entry in split_molecule(Chem.MolFromSmiles(entry))]
            for frag in frags:
                if frag in frag_vocab:
                    frag_freq[frag] += 1
        except:
            continue

    return frag_freq


def unused_vocab(file_name, smiles_col, vocab):
    df = pd.read_csv(file_name)
    unused = set()
    for smi in df[smiles_col]:
        splits = re.findall(SMI_REGEX_PATTERN, smi)
        for token in splits:
            if token not in vocab:
                unused.add(token)
    return unused


def save_vocab_to_file(vocab, file_name):
    with open(file_name, 'w') as f:
        for word in vocab:
            f.write(word + '\n')


def load_vocab_from_file(file_name):
    with open(file_name, 'r') as f:
        return [line.strip() for line in f.readlines()]


def frags_above_freq_thresh(frag_freq, threshold):
    frag_vocab = []
    for key, value in frag_freq.items():
        if value >= threshold:
            frag_vocab.append(key)
    return frag_vocab


def get_single_encoder(vocab, additional_model_tokens=None):
    """
        Additional tokens can be added for model_str2num. This may be useful for
        adding additional task-specific tokens like in MTL-BERT.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
        '<SEP>': 4,  # if fragment cannot be directly encoded, separate fragment's SMILES encoding
    }

    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num)

    vocab_str2num = {}
    for i, j in enumerate(vocab):
        vocab_str2num[j] = len(vocab_str2num) + i

    return model_str2num, vocab_str2num


def get_hybrid_encoders(smiles_vocab, frag_vocab, additional_model_tokens=None):
    """
        If there's no need for fragment encoder, input frag_vocab as None.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
        '<SEP>': 4,  # if fragment cannot be directly encoded, separate fragment's SMILES encoding
    }

    smiles_str2num = {}
    for i, j in enumerate(smiles_vocab):
        smiles_str2num[j] = len(model_str2num) + i

    if frag_vocab is not None:
        frag_str2num = {}
        for i, j in enumerate(frag_vocab):
            frag_str2num[j] = len(model_str2num) + len(smiles_vocab) + i
    else:
        frag_str2num = None

    if frag_vocab is not None:
        for token in additional_model_tokens:
            model_str2num[token] = len(model_str2num) + len(smiles_vocab) + len(frag_vocab)
    else:
        for token in additional_model_tokens:
            model_str2num[token] = len(model_str2num) + len(smiles_vocab)

    return model_str2num, smiles_str2num, frag_str2num


def get_decoders(*args):
    return [{i: j for j, i in entry.items()} for entry in args]


def smiles_encode_molecule(smi, model_str2num, smiles_str2num):
    """
        Encodes molecule using SMILES. If atom not in vocab, use <UNK>
    """
    return [smiles_str2num.get(char, model_str2num['<UNK>']) for char in re.findall(SMI_REGEX_PATTERN, smi)]


def smiles_decode_molecule(encoded_smi, model_num2str, smiles_num2str):
    """
        Decodes SMILES-encoded molecule.
    """
    res = []
    for num in encoded_smi:
        if num in model_num2str:
            res.append(model_num2str[num])
        elif num in smiles_num2str:
            res.append(smiles_num2str[num])
        else:
            raise Exception(f"Unexpected encoding value: {num}, not found in model_num2str or smiles_num2str")
    return res


def fragment_encode_molecule(smi, model_str2num, frag_str2num):
    encoded = []
    fragments = [Chem.MolToSmiles(entry) for entry in split_molecule(Chem.MolFromSmiles(smi))]

    for fragment in fragments:
        encoded.append(frag_str2num.get(fragment, model_str2num['<UNK>']))

    return encoded


def combine_vocabs(*args):
    """
    Combine multiple vocabularies, but without any duplicates.

    Note: originally, I used a python set to remove duplicates, however
    since a set is unordered, the combined vocabularies would be in a
    different order every time they're combined. Now, when combining
    the vocabularies they will always be combined in the same order.
    """
    vocab = []
    for arg in args:
        for val in arg:
            if val not in vocab:
                vocab.append(val)
    return vocab


def hybrid_encode_molecule(smi, model_str2num, smiles_str2num, frag_str2num):
    """
        Encodes a molecule using a hybrid SMILES-fragment technique.

        If molecule cannot be fragmentized, then return SMILES encoding. Otherwise
        go through fragments and if fragment not in vocabulary, use SMILES encoding.

        Note: <SEP> token is used to separate SMILES encodings between fragments so
        after decoding from the model, a list of fragments can be generated and then
        constructed back into a molecule.

        e.g. if CCO[Yb] is not in fragment vocabulary, encode
        using smiles -> C C O [Yb]. However, if CCO[Yb] appears
        next again in molecule fragments, the encoding will look
        like: C C O [Yb] C C O [Yb], thus we need
        C C O [Yb] <SEP> C C O [Yb]. This informs of the separation.
        Afterwards we know the fragments are [..., CCO[Yb], CCO[Yb], ...]
    """
    try:
        encoded = []
        fragments = [Chem.MolToSmiles(entry) for entry in split_molecule(Chem.MolFromSmiles(smi))]
        for fragment in fragments:
            if fragment in frag_str2num:
                if len(encoded) > 0 and encoded[-1] == model_str2num['<SEP>']:
                    # No need for <SEP> if next fragment is in frag_vocab
                    del encoded[-1]
                encoded.append(frag_str2num[fragment])
                # encoded.append(fragment)
            else:
                # encoded += [char for char in re.findall(SMI_REGEX_PATTERN, fragment)] + [model_str2num['<SEP>']]
                encoded.extend([smiles_str2num.get(char, model_str2num['<UNK>']) for char in
                                re.findall(SMI_REGEX_PATTERN, fragment)] + [model_str2num['<SEP>']])
        if len(encoded) > 0 and encoded[-1] == model_str2num['<SEP>']:
            # No need for <SEP> at end of encoding
            del encoded[-1]
        
        return encoded
    except Exception as e:
        #print(f"Can't hybrid encode {smi}: {e}")
        return smiles_encode_molecule(smi, model_str2num, smiles_str2num)


def hybrid_decode_molecule(encoded_smi, model_num2str, smiles_num2str, frag_num2str):
    """
        Decodes a molecule using a hybrid SMILES-fragment technique.
    """
    decoded = []
    idx = 0
    while idx < len(encoded_smi):
        token = encoded_smi[idx]
        if token in model_num2str:
            decoded.append(model_num2str[token])
        elif token in frag_num2str:
            decoded.append(frag_num2str[token])
        else:
            tmp = []
            while idx < len(encoded_smi) and encoded_smi[idx] in smiles_num2str:
                token = encoded_smi[idx]
                tmp.append(smiles_num2str[token])
                idx += 1

            if idx < len(encoded_smi):
                idx -= 1

            decoded.append("".join(tmp))
        idx += 1

    return decoded


def smiles_to_svg(smi, file_name, size=(300, 300)):
    m = Chem.MolFromSmiles(smi)
    k = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    k.DrawMolecule(m)
    k.FinishDrawing()
    with open(file_name, 'w') as f:
        f.write(k.GetDrawingText())


def convert_encoding_to_tokens(encoded_smi, model_num2str, smiles_num2str, frag_num2str):
    res = []
    for num in encoded_smi:
        num = int(num)
        if num in model_num2str:
            res.append(model_num2str[num])
        elif num in frag_num2str:
            res.append(frag_num2str[num])
        elif num in smiles_num2str:
            res.append(smiles_num2str[num])
        else:
            raise Exception(f"Unknown encoding value: {num}")
    return res