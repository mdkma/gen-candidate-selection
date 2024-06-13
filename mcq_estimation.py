import transformers
import torch
from transformers import pipeline
import torch.nn.functional as F
import json
import time 
import numpy as np
import argparse
import random 
import os 
import datetime 
from estimators import Estimator, DenseRetrieval
from metric import accuracy, precision, recall, f1_score 
from data_loader import load_data 
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0 )
parser.add_argument('--seed', type=int, default=0 )
parser.add_argument('--dataset', type=str, default='commonsense_qa', help='dataset name' )
 
parser.add_argument('--model_name', type=str, default='gpt2', help='pre-trained language model' )
parser.add_argument('--batch_size', type=int, default=1 )
parser.add_argument('--chat_template', type=int, default=0 )
parser.add_argument('--topk', type=int, default=1 )


args = parser.parse_args()
def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

manual_seed(args.seed)
 
 
device = torch.device(f"cuda:{args.cuda_id}")
 

def main(args):  
    print(args) 
    prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes = load_data(args)
    print('loaded data')

    true = []
    pred_first = []
    pred_last = []
    pred_avg = []
    pred_sample_avg = []
    pred_sum = []
    total_rt = 0
    with torch.no_grad():  
        t1 = time.time()  
        for i in tqdm.tqdm(range(len(batch_sizes))): 
            i1 = int(np.sum(batch_sizes[:i]))
            i2 = int(np.sum(batch_sizes[:i+1])) 
            choice_first, choice_last, choice_avg, choice_sample_avg, choice_sum = estimator(prompt_lis[i1 :i2], candidate_pool_lis[i1 :i2] )
            true.append(torch.tensor(answer_lis[i1 :i2]))
            pred_first.append(choice_first.detach().cpu()) 
            pred_last.append(choice_last.detach().cpu()) 
            pred_avg.append(choice_avg.detach().cpu()) 
            pred_sample_avg.append(choice_sample_avg.detach().cpu()) 
            pred_sum.append(choice_sum.detach().cpu())  
        t2 = time.time()
        total_rt += t2 - t1 
    true = torch.cat(true)
    pred_first = torch.cat(pred_first)
    pred_last = torch.cat(pred_last)
    pred_avg = torch.cat(pred_avg)
    pred_sample_avg = torch.cat(pred_sample_avg)
    pred_sum = torch.cat(pred_sum)

    today =str(datetime.datetime.now()).split(' ')[0] 

    if not os.path.exists(f"results/{today}/{args.dataset}"):
        os.makedirs(f"results/{today}/{args.dataset}")
    if '/' in args.model_name:
        model_name = args.model_name.split('/')[-1]
    else:
        model_name =  args.model_name
    method_lis = ['first', 'last', 'avg', 'sample_avg', 'sum']
    pred_lis = [pred_first, pred_last, pred_avg, pred_sample_avg, pred_sum]
    for i in range(len(method_lis)):
        pred = pred_lis[i]
        method = method_lis[i]
        res = np.array([accuracy(true, pred),
                    precision(true, pred),
                    recall(true, pred),
                    f1_score(true, pred),
                    total_rt,
                    len(prompt_lis) ])
        print(res) 
        np.savetxt(f"./results/{today}/{args.dataset}_{model_name}_{method}_seed{args.seed}.txt", 
                res.reshape(1,-1), 
                    header='accuracy,precision,recall,f1,rt,n_sample',
                    delimiter=',',
                    comments='') 
        np.save(f"./results/{today}/{args.dataset}/{model_name}_{method}_true.npy", true.cpu().numpy())
        np.save(f"./results/{today}/{args.dataset}/{model_name}_{method}_pred.npy", pred.cpu().numpy())

def main_dpr(args):  
    print(args) 
    prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes = load_data(args)
    print('loaded data')

    true = []
    pred = [] 
    total_rt = 0
    with torch.no_grad():  
        t1 = time.time()  
        for i in tqdm.tqdm(range(len(batch_sizes))): 
            i1 = int(np.sum(batch_sizes[:i]))
            i2 = int(np.sum(batch_sizes[:i+1])) 
            choice = estimator(prompt_lis[i1 :i2], candidate_pool_lis[i1 :i2] )
            true.append(torch.tensor(answer_lis[i1 :i2]))
            pred.append(choice.detach().cpu())  
             
        t2 = time.time()
        total_rt += t2 - t1 
    true = torch.cat(true)
    pred = torch.cat(pred) 
    method = 'dpr'
    today =str(datetime.datetime.now()).split(' ')[0] 

    if not os.path.exists(f"results/{today}/{args.dataset}"):
        os.makedirs(f"results/{today}/{args.dataset}")
    if '/' in args.model_name:
        model_name = args.model_name.split('/')[-1]
    else:
        model_name =  args.model_name   
    res = np.array([accuracy(true, pred),
                precision(true, pred),
                recall(true, pred),
                f1_score(true, pred),
                total_rt,
                len(prompt_lis) ])
    print(res) 
    np.savetxt(f"./results/{today}/{args.dataset}_{model_name}_{method}_seed{args.seed}.txt", 
            res.reshape(1,-1), 
                header='accuracy,precision,recall,f1,rt,n_sample',
                delimiter=',',
                comments='') 
    np.save(f"./results/{today}/{args.dataset}/{model_name}_{method}_true.npy", true.cpu().numpy())
    np.save(f"./results/{today}/{args.dataset}/{model_name}_{method}_pred.npy", pred.cpu().numpy())
        
          
   
if args.model_name == 'dpr':
    estimator = DenseRetrieval(args, device)
    main_dpr(args)  
else:  
    estimator = Estimator(args, device)  
    main(args)



 