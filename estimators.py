# Implement estimation method 
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForPreTraining 
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import random 
 

class Estimator(nn.Module):
    def __init__(self, args, device ):
        super().__init__()
        model_name = args.model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, padding_side='left')

        if model_name[:6] == 'google': 
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name) 
        self.tokenizer.pad_token_id = self.model.config.eos_token_id 
        self.model.eval()
        self.model.to(device)
        self.device = device 
  
        self.softmax = torch.nn.Softmax(dim=-1) 
        print(f'Loading pre-trained model {args.model_name}')
 
        self.args = args

    
    def get_logits(self, prompt):
        if self.args.chat_template == 1:
            prompt_ = []
            for p in prompt:
                user_message_cut = f"Question: {p} "
                # print(user_message_cut)
                if self.args.model_name == 'mistralai/Mistral-7B-Instruct-v0.3':
                    messages_this = [ 
                    {"role": "user", "content": self.args.prompt_system_role + '\n' + self.args.prompt_task_instruction + user_message_cut  }
                ] 
                else:
                    messages_this = [
                        {"role": "system", "content": self.args.prompt_system_role + '\n' + self.args.prompt_task_instruction},
                        {"role": "user", "content": user_message_cut  }
                    ] 
                
                prompt_.append(self.tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True)) 
            prompt = prompt_

        if self.args.model_name[:6] == 'google': 
            input_ids = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)

            decoder_start_token = self.tokenizer.pad_token_id if self.model.config.pad_token_id is not None else self.model.config.decoder_start_token_id
            decoder_input_ids = torch.tensor([[decoder_start_token]], dtype=torch.long).expand(len(prompt),-1).to(self.device)
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits # [batch_size, max_seq_len, voc_size] 
            logits = logits[:,-1,:] 
        else: 
            inputs = self.tokenizer(prompt, padding=True, truncation=True,return_tensors="pt")  
            

            inputs = {key: val.to(self.device) for key, val in inputs.items()}  
            outputs = self.model(**inputs)
            del inputs
            logits = outputs.logits # [batch_size, max_seq_len, voc_size] 
            logits = logits[:,-1,:] 
        return logits

    def get_topk(self, logits, candidate_pool_, n_cand):
        '''
        logits: [batch_size, voc_size] 
        '''
        batch_size=logits.shape[0] 
        cands = self.tokenizer(candidate_pool_, padding=True, truncation=True,return_tensors="pt" ) 
         
        cand_id = cands['input_ids'].to(self.device) #  (batch_size * n_cand x token_id_len)
        
        logits = torch.gather(logits, 1, cand_id.reshape(batch_size,-1)) #  batch_size x (n_cand x token_id_len)
        logits = logits.reshape(batch_size, n_cand, -1) #  batch_size x n_cand x token_id_len
        mask = cands['attention_mask'].to(self.device).reshape(batch_size, n_cand, -1)
        
        logits = logits * mask
        
        logits = logits.sum(-1) 
        prob = self.softmax(logits)
        choices = torch.argsort(prob, dim=1)[:, -self.args.topk:] # N x k
          
        return choices

        

    def get_topk_avg(self, logits, candidate_pool_, n_cand):
        batch_size=logits.shape[0] 
        cands = self.tokenizer(candidate_pool_, padding=True,  return_tensors="pt" ) 
         
        cand_id = cands['input_ids'].to(self.device) #  (batch_size x n_cand) x token_id_len
        
        logits = torch.gather(logits, 1, cand_id.reshape(batch_size, -1)) #  batch_size x (n_cand x token_id_len)
        logits = logits.reshape(batch_size, n_cand, -1) #  batch_size x n_cand x token_id_len
        mask = cands['attention_mask'].to(self.device).reshape(batch_size, n_cand, -1)
        n_token = mask.sum(-1) #  batch_size x n_cand
        logits = logits * mask
        del mask  
        logits = logits.sum(-1) #  batch_size x n_cand
        logits = logits / n_token 
        prob = self.softmax(logits)
        choices = torch.argsort(prob, dim=1)[:, -self.args.topk:] # N x k
        return choices

     
    def get_candidate_repr_single(self, candidate_pool, idx=0): 
        candidate_pool_ = [] # n_candidate
        for cand in candidate_pool:
            for j in cand:
                tks = j.strip().split(' ')
                tk = tks[idx]  
                if len(tk) > 1 and tk[-1] in ['.', '!', '?', ',', ';']:
                    tk = tk[:-1]
                candidate_pool_.append(tk)
        return candidate_pool_ 
    
    def get_candidate_repr_multi(self, candidate_pool, skip=1):  
        candidate_pool_ = [] #  n_candidate
        for cand in candidate_pool:
            for j in cand:
                tks = j.split(' ')
                sampled = tks[::skip]
                candidate_pool_.append(' '.join(sampled))  
        return candidate_pool_

    def forward(self, prompt, candidate_pool): 
        '''
        prompt: [Question 1 ... Question N]
        candidate_pool: [Option 1 ... Option k]
        
        Since all questions share the same candidate pool, we only pass one
        
        '''  
         
  
        logits = self.get_logits(prompt) 
        ##############First##################
        candidate_pool_ = self.get_candidate_repr_single(candidate_pool) 
        n_cand = len(candidate_pool[0]) 
        choice_first = self.get_topk(logits, candidate_pool_, n_cand)
        del candidate_pool_
        ##############First##################
        
        ##############Last##################
        candidate_pool_ = self.get_candidate_repr_single(candidate_pool, idx=-1) 
        choice_last = self.get_topk(logits, candidate_pool_, n_cand)
        del candidate_pool_
        ##############Last##################
         
        ##############Avg##################
        candidate_pool_ = self.get_candidate_repr_multi(candidate_pool, skip=1) 
        batch_size=logits.shape[0] 
        cands = self.tokenizer(candidate_pool_, padding=True,  return_tensors="pt" ) 
         
        cand_id = cands['input_ids'].to(self.device) #  (batch_size x n_cand) x token_id_len 
        logits_ = torch.gather(logits, 1, cand_id.reshape(batch_size, -1)) #  batch_size x (n_cand x token_id_len)
        logits_ = logits_.reshape(batch_size, n_cand, -1) #  batch_size x n_cand x token_id_len
        mask = cands['attention_mask'].to(self.device).reshape(batch_size, n_cand, -1)
        n_token = mask.sum(-1) #  batch_size x n_cand
       
        logits_ = logits_ * mask
         
        del mask, cand_id
        choice_avg = torch.argsort(self.softmax(logits_.sum(-1) / n_token), dim=1)[:, -self.args.topk:] # N x k 
        ##############Avg################## 
        ##############Sum################## 
        choice_sum = torch.argsort(self.softmax(logits_.sum(-1)), dim=1)[:, -self.args.topk:] # N x k 
        del candidate_pool_, logits_
        ##############Sum##################  

        ##############Subsampled Avg##################
        candidate_pool_ = self.get_candidate_repr_multi(candidate_pool, skip=2)  
        choice_sample_avg = self.get_topk_avg(logits, candidate_pool_, n_cand) 
        ##############Subsampled Avg################## 
     
        torch.cuda.empty_cache() 
          
        return choice_first, choice_last, choice_avg, choice_sample_avg, choice_sum 
  
class DenseRetrieval(nn.Module):
    def __init__(self, args, device):
        super().__init__()  
        args.model_name = 'facebook-dpr'
        print(args.model_name)
         
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
     
        self.question_tokenizer.pad_token_id = self.question_encoder.config.eos_token_id
        self.context_tokenizer.pad_token_id = self.context_encoder.config.eos_token_id
     
        
        self.question_encoder.eval()
        self.question_encoder.to(device)
        self.context_encoder.eval()
        self.context_encoder.to(device)
        self.device = device 
  
        self.softmax = torch.nn.Softmax(dim=-1)  
  

    def forward(self, prompt, candidate_pool, n_cand):  
        batch_size = len(prompt)
        inputs = self.question_tokenizer(prompt, padding=True, truncation=True,return_tensors="pt")  
        question_embeddings = self.question_encoder(inputs['input_ids'].to(self.device)).pooler_output 
          
        candidate_pool_ = [] # batch_size * n_candidate
        for cand in candidate_pool:
            for j in cand:
                candidate_pool_.append(j )  
         
        cands = self.context_tokenizer(candidate_pool_, padding=True, truncation=True,return_tensors="pt" ) 
         
        cand_embeddings = self.context_encoder(cands['input_ids'].to(self.device)).pooler_output
        

        cand_embeddings = cand_embeddings.reshape(question_embeddings.shape[0], -1, question_embeddings.shape[-1])

        question_embeddings = question_embeddings.unsqueeze(1).expand(-1,n_cand,-1)
        
        dot_product = torch.sum(question_embeddings * cand_embeddings, dim=-1)  # Shape: (batch_size, n_candidate)

        norm_X = torch.norm(question_embeddings, p=2, dim=-1)  # Shape: (batch_size, n_candidate)
        norm_Y = torch.norm(cand_embeddings, p=2, dim=-1)  # Shape: (batch_size, n_candidate)

        cosine_similarity = dot_product / (norm_X * norm_Y)  # Shape: (batch_size, n_candidate)
        prob = self.softmax(cosine_similarity)
        choices = torch.argsort(prob, dim=1)[:, -self.args.topk:]
        return choices 
     