import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

from GPT_modules import GPTConfig
from transformers import EsmTokenizer, EsmModel
from Tokenize_modules import sequence_to_input
from Tokenize_modules import Vocabulary, PeptideTokenizer, locate_specials, locate_non_standard_AA

vocab = Vocabulary(file_name = os.path.join('../../data','vocab/vocab.txt'))
gpt_conf = GPTConfig(voc = vocab)
ESM_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_features(data, feature_dic):
    encoded = []
    for token in data:
        encoded.append(torch.as_tensor(feature_dic[token][0], dtype=torch.float32))
    return torch.stack(encoded)

class RegressionModel(nn.Module):
    def __init__(self, hidden_feat, pooling = None, pretrained_model='../../MIC_predictor/pepESM_90_50K'):
        super(RegressionModel, self).__init__()
        self.bert = EsmModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
        self.genome_linear = nn.Sequential(nn.Linear(340, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128))
        
        self.peptide_linear = nn.Sequential(nn.Linear(480, 256))
        
        if hidden_feat == None:
            self.linear = nn.Sequential(nn.Linear(384, 1))
        else:
            self.linear = nn.Sequential(nn.Linear(384, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, 1))
        
    def forward(self, input_ids, attention_mask, genome_feat):
        embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        if self.pooling == 'mean':
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif self.pooling == 'sum': #'sum'
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1)
        elif self.pooling == 'CLS':
            hidden = embedding[:, 0, :]
            
        hidden = self.peptide_linear(hidden)
        genome_feat = self.genome_linear(genome_feat)

        in_feats = torch.cat([hidden, genome_feat], dim=1)
        predict = self.linear(in_feats)
        return predict.squeeze()

def get_reward_logp(data, predictor,genome_feature = None, invalid_reward=0.0):
    result = []
    input_ids, attention_mask = sequence_to_input(data, ESM_tokenizer)
    genome_feats = torch.cat([genome_feature for _ in range(len(data))],dim = 0)
    dataset = TensorDataset(input_ids, attention_mask,genome_feats)
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)
    predictor.eval()
    with torch.no_grad():
        preds = []
        for input_ids, attention_mask, genome_feat in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            genome_feat = genome_feat.to(device)
            output = predictor(input_ids, attention_mask,genome_feat)
            preds.append(output)
        preds = torch.cat(preds)
        preds = preds.cpu()
        preds = preds.detach().numpy()
    reward_preds = np.array([5.0 if pred <= gpt_conf.reward_thres_reg else 1.0 for pred in preds])
    return preds, reward_preds
