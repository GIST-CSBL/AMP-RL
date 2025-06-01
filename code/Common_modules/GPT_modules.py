import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch import nn, optim
from dataclasses import dataclass, asdict
from Tokenize_modules import Vocabulary, StringDataset, locate_specials, locate_non_standard_AA

@dataclass
class GPTConfig:
    ## Hyperparameter
    voc : Vocabulary
    device : str = 'cuda:0'
    d_model : int = 768
    nhead : int = 12
    num_layers : int = 12
    dropout : float = 0.1
    max_len : int = 100  
    batch_size : int = 256

    reward_thres_reg : float = 1.5
    reward_thres_hemo : float = 0.5
    reward_thres_cls : float = 0.5
  
    gamma : float = 0.97
    n_to_generate : int = 1000
    n_iterations : int = 30
    rein_opt_lr : float = 1e-4
    gen_samples : int = 100000

    target_modules = ['query', 'key', 'value', 'output']
    LoRA_r : int = 8
    LoRA_alpha : int = 16
    LoRA_dropout : float = 0.1
    lora_layer_path = None

    data_path : str = '../../data'
    genome_feature_path : str = '../../data/genomes_35/genome_features.pt'
    species_path : str = '../../data/multi_train_35_0.8.csv'

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))

class LoRALayer(nn.Module):
    def __init__(self, d_model, r):
        super(LoRALayer, self).__init__()
        self.d_model = d_model
        self.r = r
        
        self.lora_A = nn.Parameter(torch.randn(self.d_model, self.r)*0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r, self.d_model)*0.01)
    
    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)

class PositionalEncoding(nn.Module):
    """
        https://kaya-dev.tistory.com/8

        forward() returns matrix (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        
        pos = torch.arange(max_len).float().unsqueeze(dim=1)  # (1 x max_len)
        _2i = torch.arange(0, d_model, step=2).float()  # index on embedding vector dim
        
        self.encoding[:,0::2] = torch.sin(pos / (10000**(_2i/d_model)))  # even emb dim index
        self.encoding[:,1::2] = torch.cos(pos / (10000**(_2i/d_model)))  # odd emb dim index
        
    def forward(self, x):
        """ x is expected to be a batch of encoded sequences (not embedded yet) with padding """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len,:].repeat((batch_size,1,1)).to(x.device)
    
class GPTModule(nn.Module):
    """
        Note that we are using batch_first=True option.
    """
    def __init__(self, voc_size, d_model, nhead, num_layers, max_len, r, dropout=0.1, lora_layer_path = None, target_modules=None):
        super(GPTModule, self).__init__()
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.voc_size = voc_size
        self.d_model = d_model
        self.r = r
        self.max_len = max_len
        self.embedding = nn.Embedding(self.voc_size, self.d_model)
        self.posenc = PositionalEncoding(self.d_model, self.max_len)
        self.target_modules = target_modules if target_modules is not None else ['query','key','value','output']
        self.lora_layers = nn.ModuleDict()
        
        for module in self.target_modules:
            self.lora_layers[module] = nn.ModuleList([LoRALayer(self.d_model, self.r) for _ in range(self.num_layers)])

        if lora_layer_path is not None:
            self.lora_layers.load_state_dict(lora_layer_path)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dropout=self.dropout, batch_first=True)  # batch first
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.linear = nn.Linear(self.d_model, self.voc_size)
    
    def forward(self, x, memory=None, tgt_mask=None, padding_mask=None):
        """
            x.shape = (batch_size, seq_len)
            memory.shape = (batch_size, source_seq_len, d_model)
            tgt_mask.shape = (seq_len, seq_len)
            padding_mask.shape = (batch_size, seq_len)
            padding_mask.dtype = torch.bool
            
            GPT don't need memory, but we leave it as option, since memory may be used as condition for seq generation.
        """
        bs, seq_len = x.shape
        emb_x = self.embedding(x)
        psen = self.posenc(x)
        _x = emb_x + psen

        if memory is None:
            memory = torch.zeros((bs, 1, self.d_model)).to(x.device)  # source_len = 1 for memory efficiency
        if tgt_mask is None:
            tgt_mask_float = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            tgt_mask = torch.isinf(tgt_mask_float)
        if padding_mask is None:
            padding_mask = torch.zeros((bs, seq_len)).bool().to(x.device)  # consider there are no paddings

        for i in range(len(self.decoder.layers)):
            dec_x = self.decoder.layers[i]
            
            q = F.linear(_x, dec_x.self_attn.in_proj_weight[:self.d_model, :], dec_x.self_attn.in_proj_bias[:self.d_model])
            k = F.linear(_x, dec_x.self_attn.in_proj_weight[self.d_model:2*self.d_model, :], dec_x.self_attn.in_proj_bias[self.d_model:2*self.d_model])
            v = F.linear(_x, dec_x.self_attn.in_proj_weight[2*self.d_model:, :], dec_x.self_attn.in_proj_bias[2*self.d_model:])

            for j, target_module in enumerate(self.target_modules):
                if target_module == 'query':
                    q = q + self.lora_layers[target_module][i](_x)
                elif target_module == 'key':
                    k = k + self.lora_layers[target_module][i](_x)
                elif target_module == 'value':
                    v = v + self.lora_layers[target_module][i](_x)

            attn_output, _ = dec_x.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=padding_mask)
            attn_output = dec_x.self_attn.out_proj(attn_output)
            
            for target_module in self.target_modules:
                if target_module == 'output':
                    attn_output = attn_output + self.lora_layers[target_module][i](attn_output)

            _x = _x + dec_x.dropout1(attn_output)
            _x = dec_x.norm1(_x)

            memory_output, _ = dec_x.multihead_attn(_x, memory, memory, attn_mask=None, key_padding_mask=None)
            _x = _x + dec_x.dropout2(memory_output)
            _x = dec_x.norm2(_x)

            _x2 = dec_x.linear2(dec_x.dropout(dec_x.activation(dec_x.linear1(_x))))
            _x = _x + dec_x.dropout3(_x2)
            _x = dec_x.norm3(_x)
        
        return self.linear(_x)  # (batch_size, seq_len, voc_size)
    
    def save_lora_weights(self, path):
        torch.save(self.lora_layers.state_dict(), path)

    def load_lora_weights(self, path):
        self.lora_layers.load_state_dict(torch.load(path))

class LRScheduler():
    """
        This class follows the function prototypes provided by torch.optim.lr_scheduler,
        but adding check_and_step() method that gives you more control.
    """
    def __init__(self, optimizer:optim.Optimizer):
        self.optimizer = optimizer
    def step():
        raise NotImplementedError()
    def check_and_step():
        raise NotImplementedError()

class ExponentialSchedule(LRScheduler):
    """
        Keep internal counter, and only updates when step_interval is met.
    """
    def __init__(self, optimizer, multiplier:float, step_interval:int):
        super(ExponentialSchedule, self).__init__(optimizer)
        self.multiplier = multiplier
        self.step_interval = step_interval
        self.counter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.multiplier
    
    def check_and_step(self):
        self.counter += 1
        if self.counter % self.step_interval == 0:
            self.step()
    
    def get_optimizer_lrs(self):
        lrs = []
        for pg in self.optimizer.param_groups:
            lrs.append(pg['lr'])
        return lrs

class BaseGPTWrapper():
    """ This class doesn't specify training function """
    def __init__(self, config: GPTConfig, lora_layer_path = None):
        self.config = config
        self.device = config.device
        self.lora_layer_path = lora_layer_path
        self.gpt = GPTModule(config.voc.vocab_size, config.d_model, config.nhead, config.num_layers, config.max_len, dropout=config.dropout, r=config.LoRA_r, lora_layer_path = self.lora_layer_path, target_modules=config.target_modules)
        self.gpt.to(self.device)

        self.voc = config.voc
        self.bosi = self.voc.get_BEG_idx()
        self.eosi = self.voc.get_EOS_idx()
        self.padi = self.voc.get_PAD_idx()

        self.d_model = config.d_model
        self.max_len = config.max_len
        self.batch_size = config.batch_size

        self.step_counter = 0  # +1 when update with single gradient step happens

    def train_mode(self):
        self.gpt.train()
    
    def eval_mode(self):
        self.gpt.eval()

    def get_param_groups(self):
        return [{'params':self.gpt.parameters()}]

    def overwrite_emb(self, new_emb_mat: torch.Tensor, emb_grad=False):
        """
            Note that optimizers you previously used would not work for new_emb_mat.
            (old optimizer only knows the old emb_layer params)
            Please reset the optimizer to include the new_emb_mat in case you have an old optimizer running.
        """
        if self.gpt.embedding.weight.shape != new_emb_mat.shape:
            raise ValueError("Embedding dimension not same!!")
        self.gpt.embedding = nn.Embedding.from_pretrained(new_emb_mat.clone())
        self.gpt.embedding.to(self.device)
        if emb_grad:
            self.gpt.embedding.weight.requires_grad = True  # default is False

    def prob_map(self, inp_x, init_mem=None):
        """
            For given sequences of a batch (all starting with <BOS> supposedly), 
                return softmax output, probabilities for each token.
            That is, the output at i-th position is predicting probabilities at (i+1)-th position of inp_x.
            - Args:
                inp_x: (batch_size, seq_len) A batch of sequences in integer. 
            - Outputs:
                prob_map: (batch_size, vocab_size, seq_length) softmax output tensor of the given seq batch
                log_prob_map: (batch_size, vocab_size, seq_length) log-softmax output tensor of the given seq batch
        """
        bs, slen = inp_x.shape
        
        tgt_mask_float = nn.Transformer.generate_square_subsequent_mask(slen).to(self.device)
        tgt_mask = torch.isinf(tgt_mask_float)
        pad_mask = (inp_x == self.padi)

        logits = self.gpt(inp_x, memory=init_mem, tgt_mask=tgt_mask, padding_mask=pad_mask)
        probs = nn.functional.softmax(logits, dim=2)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        return probs, log_probs
        
    def unroll_target(self, tgt, custom_mask=None, init_mem=None, reward1=0, reward2=0, is_reinforce = False, gamma = 0):
        """
            For given sequences of a batch, 
                return softmax output, probabilities for the target tokens, NLL of the given sequence.
            - Args:
                tgt: (batch_size, seq_len) A batch of sequences in integer. 
                    <EOS> and <PAD> should be already in. <BOS> is not in.
                custom_mask: (batch_size, seq_len) bool tensor. If given, True positions of the matrix will 
                    edit the cloned tgt matrix content into <PAD> token. That is, those positions won't be used
                    as input for prediction. This is applied before the sequential mask.
            - Outputs:
                prob_map: (batch_size, seq_len, vocab_size) softmax output tensor of the given target batch
                likelihoods: (batch_size, seq_len) likelihood for each position at each example
                NLLLoss: (batch_size) negative log likelihood for each example. <PAD>s are ignored when calculating NLL.
        """
        bs, slen = tgt.shape
        if slen > self.gpt.max_len:
            raise ValueError("input sequence length is longer than max_len!!")
        
        tgt = tgt.to(self.device).long()
        c_tgt = tgt.clone()  # this will only be used for creating input matrix
        
        # custom_mask locates the positions not to be used for prediction
        if custom_mask is not None:
            c_tgt[custom_mask] = self.padi

        # add <BOS> token at the start, and drop the last token, since the last one won't be used for prediction input.
        vbos = torch.tensor([[self.bosi]]*bs).to(c_tgt.device)
        inp_x = torch.hstack((vbos, c_tgt[:,:-1]))  # this will maintain slen

        probs, log_probs = self.prob_map(inp_x, init_mem)
        # entropy = -torch.sum(probs * torch.log(probs + 1e-6))
        one_hot_labels = nn.functional.one_hot(tgt, num_classes=self.voc.vocab_size)
        likelihoods = (probs*one_hot_labels).sum(-1)

        # entropy_coef = initial_entropy_coef - (initial_entropy_coef - final_entropy_coef) * (episode / num_episodes)
    
        if is_reinforce == True:
            discounted_reward1 = [[score*(gamma**(idx)) for idx in range(log_probs.shape[1])] for score in reward1]
            discounted_reward2 = [[score*(gamma**(idx)) for idx in range(log_probs.shape[1])] for score in reward2]

            rewards_mean1 = np.mean(discounted_reward1)
            rewards_std1 = np.std(discounted_reward1)
            normalized_rewards1 = [(r - rewards_mean1) / (rewards_std1 + 1e-8) for r in discounted_reward1]
            rewards_mean2 = np.mean(discounted_reward2)
            rewards_std2 = np.std(discounted_reward2)
            normalized_rewards2 = [(r - rewards_mean2) / (rewards_std2 + 1e-8) for r in discounted_reward2]
            
            log_likes1 = (log_probs*one_hot_labels).sum(-1)*torch.tensor(normalized_rewards1).cuda()
            log_likes2 = (log_probs*one_hot_labels).sum(-1)*torch.tensor(normalized_rewards2).cuda()

            # The model doesn't learn the paddings.
            # let's find where tgt is <PAD>. Only <PAD>s will be True.
            padding_where = (tgt == self.padi)  # padding_where.shape = (batch_size, seq_len)
            non_pad_where = ~padding_where
    
            masked_log_likes1 = non_pad_where * log_likes1  # make <PAD> positions zero
            NLLLoss1 = -masked_log_likes1.sum(-1)
            masked_log_likes2 = non_pad_where * log_likes2  # make <PAD> positions zero
            NLLLoss2 = -masked_log_likes2.sum(-1)
            return probs, likelihoods, NLLLoss1, NLLLoss2
            
        else:
            log_likes = (log_probs*one_hot_labels).sum(-1)
            
            padding_where = (tgt == self.padi)  # padding_where.shape = (batch_size, seq_len)
            non_pad_where = ~padding_where
    
            masked_log_likes = non_pad_where * log_likes  # make <PAD> positions zero
            print_masked_log_likes = non_pad_where * (log_probs*one_hot_labels).sum(-1)
            NLLLoss = -masked_log_likes.sum(-1)
            print_NLLLoss = -print_masked_log_likes.sum(-1)
            return probs, likelihoods, NLLLoss, print_NLLLoss

    def train_n_epochs(self):
        raise NotImplementedError("BaseGPTWrapper.train_n_epochs() needs to be implemented!!")
    
    def get_ckpt_dict(self):
        ckpt_dict = self.config.dict()
        ckpt_dict.pop('voc', None)
        ckpt_dict['voc_tokens'] = self.voc.tokens  # maintain the order of tokens
        ckpt_dict['gpt_state_dict'] = self.gpt.state_dict()
        ckpt_dict['step_counter'] = self.step_counter
        return ckpt_dict
    
    @staticmethod    
    def construct_by_ckpt_dict(ckpt_dict, voc:Vocabulary, lora_layer_path = None):
        # loading GPT
        conf_dict = {}
        ckpt_dict['voc'] = voc
        for k in GPTConfig.__dataclass_fields__.keys():
            conf_dict[k] = ckpt_dict[k]
        gpt_conf = GPTConfig(**conf_dict)

        self_inst = BaseGPTWrapper(gpt_conf, lora_layer_path)
        self_inst.gpt.load_state_dict(ckpt_dict['gpt_state_dict'],strict=False)
        self_inst.step_counter = ckpt_dict['step_counter']
        return self_inst

@dataclass
class GPTGeneratorConfig:
    gpt_conf: GPTConfig
    init_lr: float = 0.0001
    lr_mult: float = 0.8  ## default just maintains the lr
    lr_decay_interval: int = 5
    ckpt_path: str = "temp{}.ckpt"  # include one placeholder position
    # The placeholder could be specified at each saving process
    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))

class GPTGenerator():
    def __init__(self, base_gpt:BaseGPTWrapper, config: GPTGeneratorConfig):
        self.config = config
        self.base_gpt = base_gpt
        self.batch_size = base_gpt.batch_size
        
        self.opt = optim.Adam(self.base_gpt.get_param_groups(), lr=config.init_lr)
        self.lr_schedule = ExponentialSchedule(self.opt, config.lr_mult, config.lr_decay_interval)
        self.prog_num = 0  # progress. This will be used at placeholder of file names.

    def train_mode(self):
        self.base_gpt.train_mode()
    
    def eval_mode(self):
        self.base_gpt.eval_mode()
    
    def sample_batch(self, msl=None, bs=None):
        """
            Sample a batch of sequences (token id format). 
            The returning sequence is in rectangular shape.
            That is, some examples will contain junk tokens after the <EOS> positions.
            - Args:
                msl: max sequence length. If None, self.base_gpt.max_len is used.
                bs: batch size to be sampled. If None, self.batch_size is used.
            - Outputs:
                cur_tgt: (batch_size, seq_length) sampled sequences.
        """
        dev = self.base_gpt.device
        voc = self.base_gpt.voc
        if msl is None: msl = self.base_gpt.max_len
        if bs is None: bs = self.batch_size

        cur_tgt = torch.full((bs,msl), voc.get_PAD_idx()).to(dev) # initially filled with padding
        padding_mask = torch.full((bs,msl), True).to(dev) # bool tensor, initialized with all paddings

        # iterate over each time step
        finished = torch.zeros(bs).byte() # memorize if the example is finished or not.
        for t in range(msl):
            padding_mask[:,t] = False  # enable one more time step for prediction
            
            # Note that unroll_target() starts by automatically adding <BOS> at the front of cloned tgt
            probs, _, _, _ = self.base_gpt.unroll_target(tgt=cur_tgt, custom_mask=padding_mask)
            tth_token = torch.multinomial(probs[:,t], num_samples=1)  # (batch_size, 1) long tensor
            cur_tgt[:,t] = tth_token.reshape(-1)
            
            EOS_sampled = (tth_token == voc.get_EOS_idx()).reshape(-1)
            finished = torch.ge(finished + EOS_sampled.cpu(), 1)
            # if all the examples have produced EOS once, we will break the loop
            if torch.prod(finished) == 1: break
        return cur_tgt

    def sample_decode(self, ssize, msl=None, bs=None):
        """ 
            Sample some sequences and return the decoded ones (string format).
            The samples which don't have EOS will be discarded.
        """
        voc = self.base_gpt.voc
        if msl is None: msl = self.base_gpt.max_len
        if bs is None: bs = self.batch_size
        generation = []
        print('========== Generation Start =========')
        while len(generation) <= ssize:
            tokens_list = self.sample_batch(msl, bs)
            EOS_exist = [] # store which sample includes EOS token
            for i in range(bs):
                if self.base_gpt.eosi in tokens_list[i]:
                    EOS_exist.append(i)
            tokens_have_EOS = tokens_list[EOS_exist,:]

            # cut off after the first <EOS>
            trunc_seq_list = voc.truncate_eos(tokens_have_EOS.cpu().numpy())
            clean_seqs = []
            for seq in trunc_seq_list:
                # spinds : special token들의 위치 반환 
                spinds = locate_specials(voc, seq)
                # spinds2 : non-standard token들의 위치 반환 
                spinds2 = locate_non_standard_AA(voc, seq)
                # special token이 없는 경우, sequence 저장
                if len(spinds) == 0 and len(spinds2) == 0 and len(seq) >= 5:
                    clean_seqs.append(seq)
            ##### cleaning part is gone...
            decoded_tokens = [voc.decode(seq) for seq in clean_seqs]
            seq_list = [''.join(tl) for tl in decoded_tokens]
            generation.extend(seq_list)
        print('========== Generation End =========')
        return generation[:ssize]



    def train_n_epochs(self, train_dataset:StringDataset, val_dataset:StringDataset, epochs:int, save_period=1, debug=None, save_path = "../../ckpt/Pretrain/Best_perplexity_pretrained_model.ckpt"):
        """ make sure you called train_mode() first, or manually setting which component to train """
        train_dldr = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_dldr = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
        epo_loss_list = []
        min_perplexity = float('inf')  # 초기 최소 perplexity 값을 무한대로 설정
        best_epoch = 0  # 최소 perplexity를 가질 때의 epoch을 기록할 변수

        for epo in range(1, epochs+1):
            print("- epoch:", epo, " - progress:", self.prog_num)
            loss_collection = 0.0
            b_cnt = 0  # count how many batches are in one epoch
            total_loss_sum = 0.0  # 전체 NLL 합
            total_length = 0  # 전체 시퀀스 길이 합

            # Training loop
            for bi, batch_data in tqdm(enumerate(train_dldr)):
                _, likelihoods, NLLLoss, _ = self.base_gpt.unroll_target(batch_data)
                mean_loss = NLLLoss.mean()

                self.opt.zero_grad()
                mean_loss.backward()
                self.opt.step()

                self.base_gpt.step_counter += 1                
                loss_collection += mean_loss.cpu().detach()
                total_loss_sum += NLLLoss.sum().cpu().detach()  # 배치의 NLL 합산
                total_length += (batch_data != 0).sum(dim=1).sum()  # 배치의 전체 단어 수
                b_cnt += 1

                if debug is not None:
                    if bi % debug == 0: 
                        print(mean_loss.cpu().detach())

            # Training loss and PPL
            epoch_loss = loss_collection / b_cnt
            epo_loss_list.append(epoch_loss)
            print("-- epoch loss:", epoch_loss)

            # Calculate true perplexity
            if total_length > 0:
                ppl = np.exp(total_loss_sum / total_length)  # 전체 데이터 기반 perplexity
            else:
                ppl = float('inf')  # 데이터가 없으면 무한대로 설정
            print("-- Training PPL:", ppl)

            # Validation loop for loss and PPL
            self.base_gpt.eval_mode()  # Switch to evaluation mode
            with torch.no_grad():
                val_loss_sum = 0.0
                val_total_length = 0

                for val_batch_data in val_dldr:
                    _, _, NLLLoss, _ = self.base_gpt.unroll_target(val_batch_data)
                    val_loss_sum += NLLLoss.sum().cpu().detach()  # 검증 데이터의 NLL 합산
                    val_total_length += (val_batch_data != 0).sum(dim=1).sum()  # 검증 데이터의 전체 단어 수

                # Calculate true validation perplexity
                if val_total_length > 0:
                    val_ppl = np.exp(val_loss_sum / val_total_length)  # 검증 데이터 기반 perplexity
                else:
                    val_ppl = float('inf')  # 데이터가 없으면 무한대로 설정

                print("-- Validation PPL:", val_ppl)

            # 최소 perplexity 비교 후 모델 저장
            if val_ppl < min_perplexity:
                min_perplexity = val_ppl
                best_epoch = epo
                self.save_ckpt_path(save_path)  # 파일명에 epoch 번호를 포함하여 저장
                print(f"New best model saved with perplexity: {min_perplexity} at epoch {epo}")

            self.base_gpt.train_mode()  # Switch back to training mode

            self.prog_num += 1
            self.lr_schedule.check_and_step()

        return epo_loss_list

    def get_ckpt_dict(self):
        ckpt_dict = self.config.dict()
        ckpt_dict.pop('gpt_conf', None)
        ckpt_dict['gpt_dict'] = self.base_gpt.get_ckpt_dict()
        ckpt_dict['opt_state_dict'] = self.opt.state_dict()
        ckpt_dict['prog_num'] = self.prog_num
        return ckpt_dict
    
        
    def save_ckpt_path(self, saveto):
        ckpt_dict = self.get_ckpt_dict()
        saveto = saveto
        print("model saved to: ", saveto)
        print("Epoch : " , self.prog_num)
        torch.save(ckpt_dict, saveto)

    @staticmethod
    def construct_by_ckpt_dict(gpt_conf, ckpt_dict, voc:Vocabulary, lora_layer_path = None):
        """ 
            please check prog_num value after construct (caution for overwriting existing ckpt file)
        """
        # build base gpt
        gpt_ckpt_dict = ckpt_dict['gpt_dict']
        gpt_ckpt_dict['target_modules'] = gpt_conf.target_modules
        gpt_ckpt_dict['LoRA_r'] = gpt_conf.LoRA_r
        gpt_ckpt_dict['LoRA_alpha'] = gpt_conf.LoRA_alpha
        gpt_ckpt_dict['LoRA_dropout'] = gpt_conf.LoRA_dropout
        base_gpt = BaseGPTWrapper.construct_by_ckpt_dict(gpt_ckpt_dict, voc, lora_layer_path)
        ckpt_dict['gpt_conf'] = base_gpt.config

        # build self
        self_conf_dict = {}
        for k in GPTGeneratorConfig.__dataclass_fields__.keys():
            self_conf_dict[k] = ckpt_dict[k]
        self_conf = GPTGeneratorConfig(**self_conf_dict)
        self_inst = GPTGenerator(base_gpt, self_conf)
        #self_inst.opt.load_state_dict(ckpt_dict['opt_state_dict'])
        self_inst.prog_num = ckpt_dict['prog_num']
        return self_inst