import os
from features import fs_encode
import lightgbm as lgb
import numpy as np
import torch
from transformers import T5Tokenizer, T5Model,T5EncoderModel
import re
from Bio import SeqIO
from GPT_modules import GPTConfig
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel
from Tokenize_modules import sequence_to_input
from Tokenize_modules import Vocabulary, PeptideTokenizer, locate_specials, locate_non_standard_AA

vocab = Vocabulary(file_name = os.path.join('../../data','vocab/vocab.txt'))
gpt_conf = GPTConfig(voc = vocab)
ESM_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm = model_esm.to(device)

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False)
model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

model_fs = lgb.Booster(model_file="../../Hemolysis_predictor/source/models/model.fs")
model_tr = lgb.Booster(model_file="../../Hemolysis_predictor/source/models/model.transformer")

def esm_infer(seqs):
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [("tmp", d) for d in seqs]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    dataset = TensorDataset(batch_tokens)
    loader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, shuffle=False)

    with torch.no_grad():
        preds = []
        for inp_tok in loader:
            results = model_esm(inp_tok[0].to(device), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            token_representations = token_representations.detach().cpu().numpy()
            token_representations = torch.tensor([inp[1:-1,:] for inp in token_representations])
            # token_representations = token_representations[0][1:-1,:]
            preds.append(torch.tensor(token_representations.sum(axis=1)))
        preds = torch.cat(preds, axis = 0)
    # (N, 1280)
    return preds

def T5_infer(seqs):
    input_ids_list = []
    attention_mask_list = []
    new_inp = []
    for seq in seqs:
        sequences_Example = [" ".join(list(seq))]
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
        new_inp.append(sequences_Example[0])

    ids = tokenizer.batch_encode_plus(new_inp, add_special_tokens=True, padding=True)

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, shuffle=False)
    
    with torch.no_grad():
        preds = []
        for inp_ids, att_mask in loader:
            embedding = model_t5(input_ids=inp_ids, attention_mask=att_mask) # decoder_input_ids=input_ids)
            encoder_embedding = embedding.last_hidden_state[:,:-1].detach().cpu()
            preds.append(encoder_embedding.sum(axis=1))
    # For feature extraction we recommend to use the encoder embedding
    preds = torch.cat(preds, axis = 0)
    return preds

def get_data(path):
    id_seq_dict = {}
    rx = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        id_seq_dict[id] = seq
    return id_seq_dict

def encode(id_seq):
    id_encode_dict= {}
    seqs = []
    fsvec = []
    for id,seq in id_seq.items():
        seq = seq
        seqs.append(seq)
        fs_data = fs_encode(seq)
        fsvec.append(fs_data)
    
    res_esm = esm_infer(seqs)
    res_t5 = T5_infer(seqs)
    res2 = torch.cat([res_t5,res_esm], axis = 1)
    for id, (fs, res) in enumerate(zip(fsvec, res2)):
        id_encode_dict[id] = [fs,res]
    return id_encode_dict

def predict(id_encode_dict):
    ress = {}
    for id,vec in id_encode_dict.items():
        vec_fs = np.array(vec[0])[np.newaxis,:]
        vec_ts = np.array(vec[1])[np.newaxis,:]
        p1 = model_fs.predict(vec_fs).flatten()[0]
        p2 = model_tr.predict(vec_ts).flatten()[0]
        t = 0.4
        if p1>0.4 and p2>0.4:
            ress[id] = max([p1,p2])
        elif p1<0.4 and p2<0.4:
            ress[id]=min([p1,p2])
        else:
            ress[id]=np.mean([p1,p2])
    return ress

def write_res(ress_dict):
    print("write result")
    f = open("predict_results.csv","w")
    for id,ps in ress_dict.items():
        tmp = id+","+str(ps)+"\n"
        f.write(tmp)
    f.close()

def classify_Hemo(seqs, path = './temp.txt'):
    with open(path, 'w') as f:
        for i, seq in enumerate(seqs):
            f.writelines(f'>seq_{i}\n')
            f.writelines(f'{seq}\n')
        f.close()
    id_seq_dict = get_data(path)
    id_encode_dict = encode(id_seq_dict)
    ress_dict = predict(id_encode_dict)

    reward_hemo = np.array([5.0 if pred <= gpt_conf.reward_thres_hemo else 1.0 for pred in list(ress_dict.values())])
    
    return list(ress_dict.values()), reward_hemo