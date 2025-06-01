import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook
from tqdm import tqdm
from GPT_modules import GPTConfig
from transformers import EsmTokenizer, EsmModel
from Tokenize_modules import Vocabulary, StringDataset, locate_specials, locate_non_standard_AA, sequence_to_input, prepare_batch
from GPT_modules import GPTGeneratorConfig, BaseGPTWrapper, GPTGenerator

vocab = Vocabulary(file_name = os.path.join('../../data','vocab/vocab.txt'))
gpt_conf = GPTConfig(voc = vocab)
ESM_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def kl_divergence(prior_logits, agent_logits):
    kl_loss = F.kl_div(F.log_softmax(prior_logits, dim=-1), F.softmax(agent_logits, dim=-1), reduction='batchmean')
    return kl_loss

# Gradient tracker class
class GradientTracker:
    def __init__(self, model):
        self.model = model
        self.max_gradients = {}

    def update_max_grads(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                max_grad = param.grad.detach().abs().max().item()
                if name not in self.max_gradients:
                    self.max_gradients[name] = max_grad
                else:
                    self.max_gradients[name] = max(self.max_gradients[name], max_grad)
    
    def print_max_grads(self):
        for name, max_grad in self.max_gradients.items():
            print(f'Max gradient for {name}: {max_grad:.4f}')

    def clear_max_grads(self):
        self.max_gradients = {}

def early_stopping(rewards, epoch, patience=3, min_epochs=5):
    """
    Check if the last 'patience' rewards are less than the highest reward in the list.
    Additionally, early stopping is not applied during the first 'min_epochs' epochs.

    :param rewards: List of reward values.
    :param epoch: Current epoch number.
    :param patience: Number of epochs to check for early stopping condition.
    :param min_epochs: Minimum number of epochs before early stopping can be applied.
    :return: Boolean indicating whether early stopping condition is met.
    """
    # If current epoch is less than min_epochs, do not apply early stopping
    if epoch < min_epochs:
        return False
    
    # Ensure there are enough rewards to check for early stopping
    if len(rewards) < patience + 1:
        return False
    
    highest_reward = max(rewards)
    if all(reward < highest_reward for reward in rewards[-patience:]):
        return True
    
    return False

class ExperienceMemory:
    def __init__(self, capacity=5000, save_dir='./FT_RL_Ver1'):
        self.capacity = capacity
        self.buffer = []  # 시퀀스, MIC 예측값, Hemolysis 예측값을 함께 저장
        self.mic_preds = []
        self.hemo_preds = []
        self.save_dir = save_dir  # 버퍼 파일이 저장될 디렉토리 경로
        os.makedirs(self.save_dir, exist_ok=True)  # 저장 디렉토리 생성

    def add_sequences(self, new_sequences, mic_preds, hemo_preds):
        """
        새로운 시퀀스, MIC 예측값, Hemolysis 예측값을 추가.
        MIC <= 1.5 및 Hemolysis <= 0.5 조건을 만족하지 않는 데이터는 필터링 후 제외.
        중복된 시퀀스는 첫 번째 등장한 시퀀스만 유지하고 나머지는 제거.
        """
        # 새로운 데이터와 기존 데이터를 결합
        combined_sequences = self.buffer + new_sequences
        combined_mic_preds = np.concatenate((self.mic_preds, mic_preds))
        combined_hemo_preds = np.concatenate((self.hemo_preds, hemo_preds))

        combined_data = list(zip(combined_sequences, combined_mic_preds, combined_hemo_preds))

        # MIC <= 1.5 및 Hemo <= 0.5 조건에 맞지 않는 시퀀스 제거
        filtered_data = {}
        for seq, mic, hemo in combined_data:
            if mic <= 1.5 and hemo <= 0.5:  # 조건에 맞는 시퀀스만 처리
                if seq not in filtered_data:
                    filtered_data[seq] = (mic, hemo)  # 첫 번째 등장한 시퀀스만 추가

        # MIC와 Hemolysis 값 기준으로 정렬 (MIC 기준 먼저, 그다음 Hemolysis 기준)
        filtered_data = sorted(filtered_data.items(), key=lambda x: (x[1][0], x[1][1]))

        # 필터링된 데이터를 capacity에 맞게 자르기
        filtered_data = filtered_data[:self.capacity]

        # buffer, mic_preds, hemo_preds 업데이트
        self.buffer, filtered_values = zip(*filtered_data) if filtered_data else ([], [])
        self.mic_preds, self.hemo_preds = zip(*filtered_values) if filtered_values else ([], [])
        
        self.buffer, self.mic_preds, self.hemo_preds = list(self.buffer), list(self.mic_preds), list(self.hemo_preds)

        # Buffer 저장
        self.save_buffer()

    def save_buffer(self):
        """
        버퍼 데이터를 지정된 경로에 Buffer_N.csv 형식으로 저장하는 메서드.
        """
        # 이미 존재하는 버퍼 파일 확인
        existing_files = [f for f in os.listdir(self.save_dir) if f.startswith('Buffer_') and f.endswith('.csv')]
        next_file_number = len(existing_files) + 1
        filename = os.path.join(self.save_dir, f'Buffer_{next_file_number}.csv')

        # CSV 파일로 저장
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sequence', 'MIC_Pred', 'Hemo_Pred'])  # 헤더 작성
            writer.writerows(zip(self.buffer, self.mic_preds, self.hemo_preds))

        print(f"Buffer saved to {filename}")

    def get_buffer(self, sample_size=500, times = 2):
        """
        현재 버퍼에 저장된 시퀀스 및 예측값 반환.
        """
        buffer_size = len(self.buffer)

        top_n = sample_size * times

        # 버퍼의 크기가 top_n보다 작으면 버퍼 전체를 사용
        if top_n > buffer_size:
            top_n = buffer_size

        # 상위 top_n개의 데이터 선택
        top_indices = list(range(top_n))

        # top_n 데이터에서 sample_size만큼 랜덤하게 선택
        if sample_size > top_n:
            sample_size = top_n
        indices = np.random.choice(top_indices, sample_size, replace=False)

        # 동일한 인덱스를 사용하여 시퀀스, mic_preds, hemo_preds를 샘플링
        sampled_buffer = [self.buffer[i] for i in indices]
        sampled_mic_preds = [self.mic_preds[i] for i in indices]
        sampled_hemo_preds = [self.hemo_preds[i] for i in indices]

        return sampled_buffer, sampled_mic_preds, sampled_hemo_preds

class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward1, get_reward2, rein_opt_lr, genome_feats = None):
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward1 = get_reward1
        self.get_reward2 = get_reward2
        self.learning_rate = rein_opt_lr
        self.nadir = np.zeros((2,))
        self.optimizer = torch.optim.Adam(self.generator.base_gpt.gpt.parameters(), lr = self.learning_rate, weight_decay = 1e-4)
        self.nadir_slack = 1.1
        self.genome_feats = genome_feats

    def update_nadir_point(self, reward_list):
        reward_mean_list = [data.mean() for data in reward_list]
        self.nadir = float(np.max(reward_mean_list) * self.nadir_slack + 1e-8)
        
        
    def policy_gradient(self, train_data_loader, prior , epoch , kl_coef , gamma, gradient_tracker ,grad_clipping=None ,**kwargs):
        vocab = self.generator.base_gpt.voc
        
        rl_loss = 0
        
        loss_collection = []
        reward_collection = []
        MIC_reward_collection = []
        Hemo_reward_collection = []
        
        for step, batch in tqdm(enumerate(train_data_loader)):
            reward = 0
            loss_Generator = 0.0
            
            seqs, _ = prepare_batch(batch, ESM_tokenizer, vocab)
            seqs = seqs.long() # converting float to long int
            _, reward1 = self.get_reward1(np.array(batch),self.predictor, self.genome_feats)
            _, reward2 = self.get_reward2(np.array(batch))

            _, _, NLLLoss1, NLLLoss2= self.generator.base_gpt.unroll_target(seqs, reward1 = reward1 , reward2 = reward2, is_reinforce = True, gamma = gamma)  

            prior_probs, _, prior_logits = prior.base_gpt.prob_map(seqs.cuda())
            agent_probs, _, agent_logits = self.generator.base_gpt.prob_map(seqs.cuda())
            
            losses_list_var = []
            reward_list_var = []
            
            condition_loss_list = []     
            
            for disc in [NLLLoss1, NLLLoss2]:
                losses_list_var.append(disc)
            
            softmax = nn.Softmax()
            
            for reward_value in [reward1, reward2]:
                reward_list_var.append(reward_value)

            self.update_nadir_point(reward_list_var)
            
            reward_mean_list = []
            for data in reward_list_var:
                reward_mean_list.append(1.0 / (data.mean()))
            
            reward_mean_list = softmax(torch.tensor(reward_mean_list))
                     
            for i, (reward, loss) in enumerate(zip(reward_mean_list, losses_list_var)):
                loss_Generator += reward.item() * loss.mean()
            
            loss_Generator += kl_coef * kl_divergence(prior_logits,agent_logits)
            
            self.optimizer.zero_grad()
            loss_Generator.requires_grad_(True)
            loss_Generator.backward()
            
            if grad_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.generator.base_gpt.gpt.parameters(), grad_clipping)
            
            gradient_tracker.update_max_grads()
            self.optimizer.step()
            
            loss_collection.append(loss_Generator.cpu().detach())
            MIC_reward_collection += reward1.tolist()
            Hemo_reward_collection += reward2.tolist()
            
        MIC_mean_reward = sum(MIC_reward_collection) / len(MIC_reward_collection)
        Hemo_mean_reward = sum(Hemo_reward_collection) / len(Hemo_reward_collection)
        mean_reward = (MIC_mean_reward + Hemo_mean_reward) / 2
        
        epoch_loss = sum(loss_collection) / len(train_data_loader)
        
        return epoch_loss, mean_reward, MIC_mean_reward, Hemo_mean_reward 