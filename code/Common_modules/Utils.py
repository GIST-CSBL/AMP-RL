import os
import torch
import torch.nn as nn
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from transformers import EsmTokenizer, EsmModel
from GPT_modules import GPTConfig
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel
from MIC_predictor import get_reward_logp
from Hemolysis_predictor import classify_Hemo
from Tokenize_modules import sequence_to_input
from Tokenize_modules import Vocabulary, PeptideTokenizer, locate_specials, locate_non_standard_AA

vocab = Vocabulary(file_name = os.path.join('../../data','vocab/vocab.txt'))
gpt_conf = GPTConfig(voc = vocab)
ESM_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
prefix_MIC  = pd.read_csv('../../data/Prefix_list.csv')
Prefix_Pred = prefix_MIC['Escherichia coli'].tolist()

# Set seed for reproducability
def set_randomness(random_seed = 2021):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

def plot_hist(prediction, n_to_generate, prefix_pred = Prefix_Pred,save_path= None):
    percentage_in_threshold = np.sum((prediction <= gpt_conf.reward_thres_reg))/len(prediction)
    print("Percentage of predictions within drug-like region:", percentage_in_threshold)
    print("Proportion of valid peptides:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, label = 'RL MIC Pred', shade=True)
    ax = sns.kdeplot(prefix_pred, label = 'Prefix MIC Pred' , shade=True)
    ax.set(xlabel='Predicted MIC', 
           title='Distribution of predicted MIC for generated molecules')
    if save_path is not None:
        plt.savefig(save_path)
    plt.legend()
    plt.show()
    
def get_classify(data, classifier, invalid_reward=0.0):
    sigmoid = nn.Sigmoid()
    result = []
    input_ids, attention_mask = sequence_to_input(data, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    output = classifier(input_ids, attention_mask)
    result.append(sigmoid(output))
    return result

def classify_AMP(classifier, seqs, tokenizer):
    sigmoid = nn.Sigmoid()
    input_ids, attention_mask = sequence_to_input(seqs, tokenizer)
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset=dataset, batch_size=gpt_conf.batch_size, num_workers=0, shuffle=False, drop_last = True)
    
    classifier.eval()
    with torch.no_grad():
        preds = []
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = classifier(input_ids, attention_mask)
            preds.append(sigmoid(output))
        preds = torch.cat(preds)
        preds = preds.cpu()
        preds = preds.detach().numpy()
    reward_classes = np.array([1.0 if pred >= gpt_conf.reward_thres_cls else 0 for pred in preds])

    return preds, reward_classes

class ClassificationModel(nn.Module):
    def __init__(self, hidden_feat, pooling = None, pretrained_model='../../MIC_predictor/pepESM_90_50K'):
        super(ClassificationModel, self).__init__()
        self.bert = EsmModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
        if hidden_feat == None:
            self.linear = nn.Sequential(nn.Linear(480, 1))
        else:
            self.linear = nn.Sequential(nn.Linear(480, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, hidden_feat),
                                        nn.ReLU(),
                                        nn.Linear(hidden_feat, 1))
        
    def forward(self, input_ids, attention_mask):
        embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        if self.pooling == 'mean':
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif self.pooling == 'sum': #'sum'
            hidden = torch.sum((embedding * attention_mask.unsqueeze(-1)), dim =1)
        elif self.pooling == 'CLS':
            hidden = embedding[:, 0, :]

        predict = self.linear(hidden).squeeze()
        return predict

def calculate_overlap_ratio(list1, list2):
    # 두 값 모두 1인 경우를 세기
    count_both_one = sum(1 for x, y in zip(list1, list2) if x == 1 and y == 1)
    
    # 전체 개수로 나누어 비율 계산 (분모가 0이 되는 것을 방지)
    total_elements = len(list1)  # 두 리스트의 길이가 같다고 가정
    if total_elements == 0:
        return 0  # 리스트가 비어있으면 비율은 0
    
    ratio = count_both_one / total_elements
    return ratio

def estimate_and_update(generator, predictor, cls_model,tokenizer,n_to_generate, genome_feat = None):
    generator.base_gpt.gpt.eval()
    with torch.no_grad():
        # sample and check the validity
        print("start sampling ...")
        generate = generator.sample_decode(ssize=gpt_conf.n_to_generate, msl=50, bs=128)
        generate = np.array(generate)
        preds,_  = get_reward_logp(generate, predictor, genome_feature = genome_feat)
        prefix_preds,_ = get_reward_logp(prefix_MIC['Sequence'].tolist(), predictor, genome_feature = genome_feat)
        classes, _ = classify_AMP(cls_model, generate, tokenizer)
        hemo_pred, _ = classify_Hemo(generate)
        classes = np.array([1 if data >= 0.5 else 0 for data in classes])
        micpreds = np.array([1 if data <= gpt_conf.reward_thres_reg else 0 for data in preds])
        nonhemo = np.array([1 if data <= gpt_conf.reward_thres_hemo else 0 for data in hemo_pred])
        plot_hist(preds, n_to_generate, prefix_pred = prefix_preds)
        print(f'The ratio of generated AMP : {classes.sum()/len(classes)}')
        print(f'The ratio of non-hemolytic generated peptides : {nonhemo.sum()/len(nonhemo)}')
        both_ratio = calculate_overlap_ratio(micpreds, nonhemo)
        print(f'The ratio of Both (Low MIC adn Low Hemolysis) peptides : {both_ratio}')
        print(f'The ratio of non-redundant peptides : {len(list(set(generate)))/len(generate)}')
    generator.base_gpt.gpt.train()
    return generate, preds

def simple_moving_average(previous_values, new_value, ma_window_size=1):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def get_num_amino_acid(seqs):
    amino_acid = {   'A': 0,
                     'C': 0,
                     'D': 0,
                     'E': 0,
                     'F': 0,
                     'G': 0,
                     'H': 0,
                     'I': 0,
                     'K': 0,
                     'L': 0,
                     'M': 0,
                     'N': 0,
                     'P': 0,
                     'Q': 0,
                     'R': 0,
                     'S': 0,
                     'T': 0,
                     'V': 0,
                     'W': 0,
                     'Y': 0,
                     ' ': 0
                 }
    for seq in seqs:
        for residue in seq:
            amino_acid[residue] += 1
    
    total = sum(amino_acid.values())
    for i in amino_acid.keys():
        amino_acid[i] /= total
    
    return amino_acid

def count_parameters(model):
    # 학습 가능한 파라미터 수와 크기 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compute_perplexity(log_likelihoods, seq_lens):
    """Computes a MC estimate of perplexity per word based on given likelihoods/ELBO.

    Args:
        log_likelihoods(list of float): likelihood or ELBO from N runs over the same data.
        seq_lens(list of int): the length of sequences in the data, for computing an average.

    Returns:
        perplexity(float): perplexity per word of the data.
        variance(float): variance of the log_likelihoods/ELBO that were used to compute the estimate.
    """
    # Compute perplexity per word and variance of perplexities in the samples
    perplexity = np.exp(np.array(log_likelihoods).mean() / np.array(seq_lens).mean())
    if len(log_likelihoods) > 1:
        variance = np.array(log_likelihoods).mean(axis=1).std(ddof=1)
    else:
        variance = 0.0

    return perplexity, variance 