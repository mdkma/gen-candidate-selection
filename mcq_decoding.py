import os, json, random, copy, re, time, pickle 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import re 
import torch  
from tqdm import tqdm
import copy
import gc 
import numpy as np 
from data_loader import load_data
import datetime
from metric import accuracy, precision, recall, f1_score 
import argparse
import tqdm   

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=0 )
parser.add_argument('--dataset', type=str, default='commonsense_qa', help='dataset name' )
parser.add_argument('--model_name', type=str, default='gpt2', help='pre-trained language model' )
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--subject', type=str, default='logical_deduction_three_objects', help='dataset name' )
args = parser.parse_args()
print(args)
def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

manual_seed(args.seed)

target_task = 'target_diagnoses'
# target_task = 'target_procedures'
# target_task = 'target_laborders'
# target_task = 'target_prescriptions'

split_to_use = 'test' 
args.method = 'full-decode'

if args.model_name == 'google/flan-t5-xl':
    inference_engine = 'hf'
    args.batch_size = 1
else:
    inference_engine = 'vllm' 
    args.batch_size = 50
llm_name = args.model_name   
llm_path = llm_name
print('llm_name', args.model_name)
 
hf_access_token = os.getenv('HF_ACCESS_TOKEN') 
from_saved = True

if 'gpt' in llm_name and '2' not in llm_name:
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'hf':
    if '8x7b' in llm_name.lower():
        batch_size = 4
        batch_size_big = 40
    elif 'Mistral-7B' in llm_name:
        batch_size = 1
        batch_size_big = 1
    elif '7b' in llm_name.lower():
        batch_size = 4
        batch_size_big = 40
    else:
        batch_size = 8
        batch_size_big = 80
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'vllm':
    batch_size_big = 100
 
save_name = llm_name.split('/')[-1] if '/' in llm_name else llm_name
 
evaldata_save_path = f'data/mimic4/{target_task}/{split_to_use}_{save_name}_evaldata.pkl'
batch_proc = False
char_per_token_est = 2 # -1, 4, 3, 2
token_count_lab = 500
 
device = f"cuda:{args.cuda_id}"
tokenizer = AutoTokenizer.from_pretrained(llm_name, use_auth_token=True, padding_side='left' )
    
 
if llm_name[:6] == 'google':
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
else:
    model = AutoModelForCausalLM.from_pretrained(llm_name) 
model = model.to(device) 
default_generation_config = model.generation_config
if llm_name[:6] == 'google':
    # https://github.com/huggingface/transformers/issues/5204
    default_max_length = 65536#512
else:
    default_max_length = model.config.max_position_embeddings
default_max_length_output = 100# max(model.generation_config.max_length, 500)
if inference_engine == 'vllm':
    print(f'Using vLLM {llm_path}, default_max_length_output is {default_max_length_output}')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=default_generation_config.temperature, 
                                     top_p=default_generation_config.top_p,
                                     max_tokens=default_max_length_output )
    llm_model = LLM(model=llm_path)  

# Set max token length for each model
assert default_max_length <= 65536, f'the max_token_length {default_max_length} is probably wrong, please double check'
max_token_length = default_max_length - 70 
 
  
def inference(batch, prompt_lis, candidate_pool_lis, verbose=False): 
    outputs = []  
    sentences = [ ]
    for i in range(len(prompt_lis)):
        prompt = prompt_lis[i]
        candidate_pool = candidate_pool_lis[i]
        res = [f'({chr(65+k)}) {candidate_pool[k]}' for k in range(len(candidate_pool))] 
        
        user_message_cut = f"What is the correct answer to this question: {prompt} " + f"Choices: {', '.join(res)}. " 
         
        messages_this = [
                {"role": "system", "content": args.prompt_system_role + '\n' + args.prompt_task_instruction},
                {"role": "user", "content": user_message_cut  }
            ] 
        if args.model_name in ['mistralai/Mistral-7B-Instruct-v0.3' , 'meta-llama/Meta-Llama-3-8B-Instruct']:
            inputs_ = tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True) 
        else:
            inputs_ = user_message_cut + '\n' + args.prompt_task_instruction 
        sentences.append(inputs_) 
        
    if inference_engine == 'hf': 
        if verbose: 
            print(sentences[0])
        try:
            inputs = tokenizer(sentences, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU
            # Generate the output sequence
            output_sequences = model.generate(**inputs, max_length=default_max_length_output, num_beams=4)
            # Decode the output sequence
            for outputs_single in output_sequences:
                output_ = tokenizer.decode(outputs_single , skip_special_tokens=True)
                  
                outputs.append(output_)
        except Exception as e:
            print(f"Failed to process {[dp for dp in prompt_lis]}: {e}")
            outputs.append([])
    elif inference_engine == 'vllm': 
        if verbose:
            print(sentences[0])
        outputs_batch = llm_model.generate(sentences, sampling_params)
        for outputs_single in outputs_batch: 
            outputs.append(outputs_single.outputs[0].text)
             
    # Detect multiple types of answer appearance 
    choices = []
    count = 0
    for o in outputs: 
        exist = False  
        for k in range(65,65 + args.n_cand):  
            occ_idx2 = o.find(f'Answer: {chr(k)}')   
            if occ_idx2 == -1: 
                occ_idx = o.find(f"({chr(k)})") 
                 
                if occ_idx == -1:
                    occ_idx1 = re.search(f' {chr(k)}' + r'[,.)]', o)   
                    if occ_idx1 is None:
                        occ_idx = np.inf 
                    else:
                        occ_idx = occ_idx1.span()[0]
                        exist = True  
                else:  
                    exist = True  
            else:
                occ_idx = occ_idx2 
                exist = True   
            if exist:
                choices.append(k-65)
                break
        if not exist: 
            choices.append(random.sample(range(args.n_cand),1)[0]) 
        count += 1
    torch.cuda.empty_cache() 
    return outputs, choices

 
 
prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes = load_data(args)
print('loaded data')

if args.model_name in ['mistralai/Mistral-7B-Instruct-v0.3' , 'meta-llama/Meta-Llama-3-8B-Instruct']:
    args.prompt_task_instruction = "The task is to choose one of the candidate choices that best answer the given question. "
else:
    args.prompt_task_instruction = "Which one of the choices best answers the given question? "
     
args.prompt_task_instruction = 'Format your response as follows: "The correct answer is (insert answer here)".'
 
args.n_cand = n_candidate_lis[0]
prompt_basic_length_char = len(args.prompt_system_role) + len(args.prompt_task_instruction)  
left_char_count = max_token_length * char_per_token_est - prompt_basic_length_char

true = []
pred = []
outputs = []
total_rt = 0
acc_str_lis = []
today =str(datetime.datetime.now()).split(' ')[0] 
if not os.path.exists(f"./results/{today}/{args.dataset}/{llm_name}_output"):
    os.makedirs(f"./results/{today}/{args.dataset}/{llm_name}_output")

with torch.no_grad():  
    t1 = time.time()  
    for i in tqdm.tqdm(range(len(batch_sizes))): 
        print(args.model_name, args.dataset)
        i1 = int(np.sum(batch_sizes[:i]))
        i2 = int(np.sum(batch_sizes[:i+1])) 
        #choices = estimator(prompt_lis[i1 :i2], candidate_pool_lis[i1 :i2], answer_lis[i1 :i2])
         
        output, choices = inference(i, prompt_lis[i1 :i2], candidate_pool_lis[i1 :i2])
        #print(choices)
        #print(answer_lis[i1 :i2])
        true.append(torch.tensor(answer_lis[i1 :i2]))
        pred.append(torch.tensor(choices))
        #print(pred[-1])  
        
        acc_str = f"Accuracy: {accuracy(torch.tensor(answer_lis[i1 :i2]), torch.tensor(choices))*100:.2f}"
        print(acc_str)
        acc_str_lis.append(acc_str)
        outputs += (output) 
 
    t2 = time.time()
    total_rt += t2 - t1

 
 
true = torch.cat(true)
pred = torch.cat(pred)
 
if not os.path.exists(f"results/{today}/{args.dataset}"):
    os.makedirs(f"results/{today}/{args.dataset}")
 
res = np.array([accuracy(true, pred),
            precision(true, pred),
            recall(true, pred),
            f1_score(true, pred),
            total_rt,
            len(prompt_lis) ])
print(res)
if '/' in llm_name:
    llm_name = llm_name.split('/')[-1]

with open(f"./results/{today}/{args.dataset}_{llm_name}_{args.method}_output_seed{args.seed}.txt", "w") as f: 
    for o in outputs:
        f.write(o)
        f.write('\n-------------------------\n')
with open(f"./results/{today}/{args.dataset}_{llm_name}_{args.method}_acc_str.txt", "w") as f: 
    for o in acc_str_lis:
        f.write(o)
        f.write('\n-------------------------\n')

with open(f"./results/{today}/{args.dataset}_{llm_name}_{args.method}_answer_seed{args.seed}.txt", "w") as f:
    for a in answer_lis:
        f.write(str(a))
        f.write('\n')
np.savetxt(f"./results/{today}/{args.dataset}_{llm_name}_{args.method}_seed{args.seed}.txt", 
           res.reshape(1,-1), 
            header='accuracy,precision,recall,f1,rt,n_sample',
            delimiter=',',
            comments='') 
        
np.save(f"./results/{today}/{args.dataset}/{llm_name}_{args.method}_true.npy", true.cpu().numpy())
np.save(f"./results/{today}/{args.dataset}/{llm_name}_{args.method}_pred.npy", pred.cpu().numpy())
