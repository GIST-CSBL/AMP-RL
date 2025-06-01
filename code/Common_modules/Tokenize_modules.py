import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader

class Vocabulary(object):
    
    special_tokens = ['<CLS>','<BEG>','<EOS>','<PAD>','<MASK>','<UNK>']
    def get_CLS_idx(self): return self.tok2id['<CLS>']
    def get_BEG_idx(self): return self.tok2id['<BEG>']
    def get_EOS_idx(self): return self.tok2id['<EOS>']
    def get_PAD_idx(self): return self.tok2id['<PAD>']
    def get_MASK_idx(self): return self.tok2id['<MASK>']
    def get_UNK_idx(self): return self.tok2id['<UNK>']

    def __init__(self, list_tokens=None, file_name=None, max_length=500):
        """
            If file doesn't contain one of the special tokens, 
            we manually add the token to the end of self.tokens.
        """
        if list_tokens is None and file_name is None:
            print("Please specify list_tokens or file_name !!")
            return
        if file_name is not None:
            with open(file_name, 'r') as f:
                list_tokens = [line.strip() for line in f.readlines()]
        for spc in self.special_tokens:
            if spc not in list_tokens:
                list_tokens.append(spc)
        self.init_vocab(list_tokens, max_length)

    def init_vocab(self, list_tokens, max_length):
        self.tokens = list_tokens
        self.vocab_size = len(self.tokens)
        self.tok2id = dict(zip(self.tokens, range(self.vocab_size)))
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        self.max_length = max_length

    def have_invalid_token(self, token_list):
        for i, token in enumerate(token_list):
            if token not in self.tok2id.keys():
                return True
        return False

    def encode(self, token_list):
        """Takes a list of tokens (eg ['C','(','Br',')']) and encodes to array of indices"""
        if type(token_list) != list:
            print("encode(): the input was not a list type!!!")
            return None
        idlist = np.zeros(len(token_list), dtype=np.int32)
        for i, token in enumerate(token_list):
            try:
                idlist[i] = self.tok2id[token]
            except KeyError as err:
                print("encode(): KeyError occurred! %s"%err)
                raise
        return idlist
        
    def decode(self, idlist):
        """Takes an array of indices and returns the corresponding list of tokens"""
        return [self.id2tok[i] for i in idlist]

    def truncate_eos(self, batch_seqs:np.ndarray):
        """
            This function cuts off the tokens(id form) after the first <EOS> in each sample of batch.
            - Input: batch of token lists np.ndarray(batch_size x seq_len)
            - Output: truncated sequence list
        """
        bs, _ = batch_seqs.shape
        seq_list = []
        for i in range(bs):
            ids = batch_seqs[i].tolist()
            # append EOS at the end
            ids.append(self.get_EOS_idx())
            # find EOS position of first encounter
            EOS_pos = ids.index(self.get_EOS_idx())
            # get the seq until right before EOS
            seq_list.append(ids[0:EOS_pos])
        return seq_list

    def locate_specials(self, seq):
        """Return special (BOS, EOS, PAD, or any custom special) positions in the token id sequence"""
        spinds = [self.tok2id[spt] for spt in self.special_tokens]
        special_pos = []
        for i, token in enumerate(seq):
            if token in spinds:
                special_pos.append(i)
        return special_pos

class PeptideTokenizer(object):
    def __init__(self, vocab_obj: Vocabulary):
        self.vocab_obj = vocab_obj
        # multi_chars는 cl과 같은 multiple char set이다.
        self.multi_chars = set()
        for token in vocab_obj.tokens:
            if len(token) >= 2 and token not in vocab_obj.special_tokens:
                self.multi_chars.add(token)
    
    def tokenize(self, sequence):
        # 문자열을 list로 변환 ex) ['ATGG']
        token_list = [sequence]
        
        for k_token in self.multi_chars:
            new_tl = []
            for elem in token_list:
                sub_list = []
                splits = elem.split(k_token)
                for i in range(len(splits)-1):
                    sub_list.append(splits[i])
                    sub_list.append(k_token)
                sub_list.append(splits[-1])
                new_tl.extend(sub_list)
            token_list = new_tl
        new_tl = []
        for token in token_list:
            if token not in self.multi_chars:
                new_tl.extend(list(token))
        return new_tl
    
def locate_specials(vocab: Vocabulary, seq):
    """Return special token (BEG, EOS, PAD) positions in the token sequence"""
    spinds = [vocab.get_BEG_idx(), vocab.get_EOS_idx(), vocab.get_PAD_idx(), vocab.get_MASK_idx(), vocab.get_CLS_idx(), vocab.get_UNK_idx()]
    special_pos = []
    for i, token in enumerate(seq):
        if token in spinds:
            special_pos.append(i)
    return special_pos

def locate_non_standard_AA(vocab: Vocabulary, seq):
    """Return non-standard amino acid positions in the token sequence"""
    spinds = [vocab.tok2id['B'], vocab.tok2id['Z'], vocab.tok2id['U'], vocab.tok2id['X'], vocab.tok2id['O']]
    non_standard_AA_pos = []
    for i, token in enumerate(seq):
        if token in spinds:
            non_standard_AA_pos.append(i)
    return non_standard_AA_pos

def get_valid_peptide(peptides):
    valid = []
    invalid_ids = []
    for i, sequence in enumerate(peptides):
        if '<PAD>' in sequence or '<BEG>' in sequence or '<EOS>' in sequence or '<MASK>' in sequence:
            invalid_ids.append(sequence)
        else:
            valid.append(sequence)
    return valid, invalid_ids

def collate_fn(arr, PAD_idx):
    """Function to take a list of encoded sequences and turn them into a batch"""
    # batch를 이용하기 위해 max length를 계산
    max_length = max([seq.size for seq in arr])
    # torch.full : 똑같은 값으로 해당 shape 만큼 PAD_idx 값으로 채워넣기
    collated_arr = torch.full((len(arr), max_length), PAD_idx, dtype=torch.float32)
    # 원래의 데이터로 collated_arr에 넣어주기
    for i, seq in enumerate(arr):
        collated_arr[i, :seq.size] = torch.Tensor(seq)
    return collated_arr

def prepare_batch(sequence_list, tokenizer, vocab_obj):
    EOS_idx = vocab_obj.get_EOS_idx()
    PAD_idx = vocab_obj.get_PAD_idx()
    
    # tokenizer.tokenize(sequence)는 list형태로 반환 됨 -> sample_batch_t는 이중 리스트 구조임
    sample_batch_t = [tokenizer.tokenize(sequence) for sequence in sequence_list]
    # sample_batch_e는 모든 sequence가 encode되고 난 결과를 append 한 list
    sample_batch_e = []
    keyerror_ids = []
    
    for i, tokens in enumerate(sample_batch_t):
        try:
            encoded = vocab_obj.encode(tokens)
        except KeyError as err:
            keyerror_ids.append(i)
            print(tokens)
            print("KeyError at %s"%sequence_list[i])
            continue
        sample_batch_e.append(encoded)
    # add <EOS> at the end
    EOS_batch = []
    for tokens in sample_batch_e:
        tokens = list(tokens)
        tokens.append(EOS_idx)
        EOS_batch.append(np.array(tokens, dtype=np.float64))
    # pad each example to the length of the longest in the batch
    sample_batch = collate_fn(EOS_batch, PAD_idx)
    return sample_batch, keyerror_ids

def sequence_to_input(data, tokenizer):
    
    seqs = data
    input_seqs = []
    # 단어 중간중간 ' '를 넣어주는 작업
    for i in seqs:
        for j in range(len(i)-1):
            i = i[:j+j+1]+ ' ' + i[j+j+1:]
        input_seqs.append(i)
    
    inputs = tokenizer.batch_encode_plus(input_seqs, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(inputs['input_ids'])
    attention_mask = torch.tensor(inputs['attention_mask'])
    
    return input_ids, attention_mask

def data_loader(data, tokenizer, label, BATCH_SIZE, NUM_THREADS, shuffle=False):
    input_ids, attention_mask = sequence_to_input(data, tokenizer)
    dataset = TensorDataset(input_ids, attention_mask, label)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=NUM_THREADS, shuffle=shuffle)
    
    return loader

class StringDataset(Dataset):
    """
        This is initialized with vocab, and list of strings.
        The strings will be tokenized and encoded by the vocab when __getitem__ is called.
        As tokenizing, we add EOS token at the end of the sequence.
    """
    def __init__(self, voc:Vocabulary, peptide_tokenizer, strings):
        self.voc = voc
        self.tokenizer = peptide_tokenizer
        self.strings = strings
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        toks = self.tokenizer.tokenize(self.strings[idx]) + [self.voc.id2tok[self.voc.get_EOS_idx()]]
        return torch.tensor(self.voc.encode(toks))

    def collate_fn(self, batch):
        string_ids_tensor = rnn.pad_sequence(batch, batch_first=True, 
                                            padding_value=self.voc.get_PAD_idx())
        return string_ids_tensor

def truncate_EOS(batch_token_array, EOS_token_idx):
    """
        The RNN model sampling produces (batch_size x seq_len) of sequences.
        This function cuts off the tokens after the first <EOS> in each sample.
        Input: batch of token lists np.array(batch_size x seq_len)
        Output: truncated sequence list
    """
    # batch_token_array : np.array(batch_size x seq_len)
    bs, _ = batch_token_array.shape
    seq_list = []
    """
        내가 추측하기로는, 모든 sequence list 마지막에 EOS 토큰 넣고, 만약 넣어준 EOS 말고 그 전에 처음으로 발견되면
        처음부터 맨 처음 발견된 EOS까지 sequence를 반환하고, 원래 sequence에 EOS가 없었다면, 처음부터 맨 끝까지를 반환
    """
    for i in range(bs):
        tokens = batch_token_array[i].tolist()
        # append EOS at the end
        tokens.append(EOS_token_idx)
        # find EOS position of first encounter
        EOS_pos = tokens.index(EOS_token_idx)
        # get the seq until right before EOS
        seq_list.append(tokens[0:EOS_pos])
    return seq_list

def decode_seq_list(batch_token_list, tokenizer):
    """
        Input: batch of token(index) lists (batch_size,)
        Output: (decoded) smiles list
    """
    bs = len(batch_token_list)
    peptide_list = []
    for i in range(bs):
        tokens = batch_token_list[i]
        peptide = tokenizer.decode(tokens)
        peptide_list.append(peptide)
    return peptide_list