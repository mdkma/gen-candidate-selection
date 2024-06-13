import os, json, random, copy, re, time, pickle 
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch 
from tqdm import tqdm
import copy 
from statistics import mean, median
import argparse
from estimators import Estimator 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0 )
parser.add_argument('--seed', type=int, default=0 )
parser.add_argument('--target-task', type=str, default='target_diagnoses', help='dataset name' )
 
parser.add_argument('--method', type=str, default='first', help='decoding estimation method' )
parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B', help='pre-trained language model' )
parser.add_argument('--batch_size', type=int, default=50 )
parser.add_argument('--chat_template', type=int, default=0 )
parser.add_argument('--topk', type=int, default=20 )


args = parser.parse_args()
def manual_seed(seed: int):
    random.seed(seed) 
    torch.manual_seed(seed)
manual_seed(args.seed)

 

target_task = args.target_task 
split_to_use = 'test'  
 
llm_name = args.model_name 
llm_path = llm_name
 
save_path_parsed = 'cli-data'
openai_engine = 'openai'  
from_saved = True


###### Load candidate pool
with open(f"{save_path_parsed}/score_cache_{target_task}_candidate_str_pool.json", 'r') as f:
    candidate_pool = json.load(f)
###### 
 
batch_size = 1 
batch_size_big = 1 
llm_name_trunc = llm_name.split('/')[-1] if '/' in llm_name else llm_name
save_name = f"{llm_name_trunc}"  
evaldata_save_path = f'cli-data/{target_task}/{split_to_use}_{save_name}_est_evaldata.pkl'
batch_proc = False
char_per_token_est = 2 # -1, 4, 3, 2
token_count_lab = 500

# Prepare instructioin and system role for the selected task
prompt_system_role = 'You are a professional clinician in a hospital with expert knowledge in medical and clinical domains.'
if target_task == 'target_diagnoses':
    prompt_task_instruction = 'The task is to make a list of diagnoses for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'The diagnosis can be in ICD-10-CM code format (such as S12.000G), or natural language description of the disease. '
    prompt_task_instruction += 'Separate each diagnosis with a new line. '
    prompt_task_instruction += 'Please provide as many diagnoses as you can until you are not confident about your diagnosis decision. '

    prompt_task_instruction_end = 'What are the diagnoses for this patient?'
elif target_task == 'target_procedures':
    prompt_task_instruction = 'The task is to decide a list of procedures for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'A clinical procedure can be defined as any practice of a health practitioner that involves a combination of special skills or abilities and may require drugs, devices, or both. Clinical procedure is an activity directed at or performed on an individual with the object of improving health, treating disease or injury, or making a diagnosis. '
    prompt_task_instruction += 'The procedure can be in ICD-10-PCS code format (such as 4A023N6), or natural language description of the procedure. '
    prompt_task_instruction += 'Separate each procedure with a new line. '
    prompt_task_instruction += 'Please provide as many procedures as you can until you are not confident about your procedure decision. '

    prompt_task_instruction_end = 'What are the procedures for this patient?'
elif target_task == 'target_laborders':
    prompt_task_instruction = 'The task is to decide a list of lab tests to be done for this patient based on the provided information of the patient to facilitate downstream diagnosis. '
    prompt_task_instruction += 'Lab test is a medical procedure that involves testing a sample of blood, urine, or other substance from the body. Laboratory tests can help determine a diagnosis, plan treatment, check to see if treatment is working, or monitor the disease over time. '
    prompt_task_instruction += 'Please produce natural language name or definition of the lab tests to be ordered. '
    prompt_task_instruction += 'Separate each lab test with a new line. '
    prompt_task_instruction += 'Please provide as many lab tests as you can until you are not confident about your lab test order decision. '

    prompt_task_instruction_end = 'What lab tests need to be ordered for this patient?'
elif target_task == 'target_prescriptions':
    prompt_task_instruction = 'The task is to decide a list of medications to be prescribed for this patient based on the provided information of the patient. '
    prompt_task_instruction += 'Please produce natural language brand names or generic names of the medications. '
    prompt_task_instruction += 'Separate each medication with a new line. '
    prompt_task_instruction += 'Please provide as many prescriptions as you can until you are not confident about your prescription decision. '

    prompt_task_instruction_end = 'What medications need to be prescribed for this patient?'
else:
    assert False, 'Not implemented for other tasks'

with open(os.path.join(save_path_parsed, 'labitem_labels.json')) as f:
    labitem_labels = json.load(f)

 
tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16 )

default_generation_config = model.generation_config
default_max_length = model.config.max_position_embeddings
default_max_length_output = max(model.generation_config.max_length, 500)
 
device = torch.device(f"cuda:{args.cuda_id}")
estimator = Estimator(args, device) 
 
assert default_max_length <= 65536, f'the max_token_length {default_max_length} is probably wrong, please double check'
max_token_length = default_max_length - 650
 
prompt_basic_length_char = len(prompt_system_role) + len(prompt_task_instruction) + len(prompt_task_instruction_end)
left_char_count = max_token_length * char_per_token_est - prompt_basic_length_char

def model_specific_prompt(system_prompt, task_instruction, user_message, task_instruction_end):
    if 'gpt' in llm_name.lower():
        return f"{task_instruction}\n{user_message} {task_instruction_end}"
    if 'llama' in llm_name.lower() or 'alpaca' in llm_name.lower():
        # not having <s> at the beginning, since it will be added by tokenizer automatically
        return f"""[INST] <<SYS>>
{ system_prompt }

{ task_instruction }
<</SYS>>

{ user_message } 
{ task_instruction_end } [/INST]"""
    if 'mistral' in llm_name.lower() or 'mixtral' in llm_name.lower():
        return f"""[INST] { system_prompt }
{ task_instruction }

{ user_message }
{ task_instruction_end } [/INST]"""

def inference(dps, verbose=False):
    responses = []
    outputs_first = []
    outputs_last = []
    outputs_avg = []
    outputs_sample_avg = []
    outputs_sum = []
    outputs_sample_sum = []

    dps_is_list = True
    if not isinstance(dps, list):
        dps = [dps]
        dps_is_list = False  
        
    sentences = [dp['input'] for dp in dps]  
    if batch_proc:
        inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
        # print(inputs['input_ids'].shape)
        output_sequences = model.generate(
            **inputs,
            do_sample=True,
            # top_k=10, # might lead to RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=default_max_length_output,
        )
        outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        if verbose: 
            print(sentences[0])
        choice_first, choice_last, choice_avg, choice_sample_avg, choice_sum = estimator(sentences, candidate_pool)
        for i in range(choice_first.shape[0]): 
            outputs_first.append('\n'.join([candidate_pool[k] for k in choice_first[i]]))
            outputs_last.append('\n'.join([candidate_pool[k] for k in choice_last[i]]))
            outputs_avg.append('\n'.join([candidate_pool[k] for k in choice_avg[i]]))
            outputs_sample_avg.append('\n'.join([candidate_pool[k] for k in choice_sample_avg[i]]))
            outputs_sum.append('\n'.join([candidate_pool[k] for k in choice_sum[i]]))

    if not dps_is_list:
        responses = responses[0]
        outputs = outputs[0]
    return outputs_first, outputs_last, outputs_avg, outputs_sample_avg, outputs_sum, outputs_sample_sum 


if from_saved and os.path.exists(evaldata_save_path):
    evaldata = pickle.load(open(evaldata_save_path, 'rb'))
    print(f'\n\n-> Using cached evaldata saved at {evaldata_save_path}\n\n')
else: 
    if split_to_use == 'train':
        evaldata = pickle.load(open(f'cli-data/{target_task}/{split_to_use}.pkl', 'rb'))
        # sample 1% of the training data
        evaldata = random.sample(evaldata, int(len(evaldata) * 0.1))
        print(f'sampled {len(evaldata)} data points')
    else:
        # with open(f'cli-data/{target_task}/{split_to_use}.json', 'r') as f:
        #     evaldata = json.load(f)
        with open(f'cli-data/{target_task}/{split_to_use}.pkl', 'rb') as f:
            evaldata = pickle.load(f)

    gendata = []

    cut_count_user_msg_end = 0
    cut_count_user_msg_note = 0
    cut_rates = []
    if 'gpt' not in llm_name.lower():
        # For open-source models, we can cut the input directly here
        prompt_basic = prompt_system_role + prompt_task_instruction + prompt_task_instruction_end
        left_token_count = max_token_length - len(tokenizer(prompt_basic)['input_ids']) - 30
        print(f'left_token_count: {left_token_count}')

    for dp_i, dp in enumerate(tqdm(evaldata, desc=f"{target_task}, {llm_name}, gen data")):
        patient_info = f"Patient information: age is {dp['patient_age']}, gender is {dp['patient_gender']}, race is {dp['patient_race']}, marital status is {dp['patient_marital']}, insurance category is {dp['patient_insurance']}, admission_location is {dp['admission_location']}. "
        
        # include cleaned discharge note for all kinds of tasks
        discharge_note = ''
        if len(dp['notes_discharge']) > 0:
            discharge_note += '\n\nCLINICAL NOTE: \n'
            for note_item in dp['notes_discharge']:
                discharge_note += note_item[0] + '\n'

        # only include radiology note for the diagnosis task
        radiology_note = ''
        if target_task == 'target_diagnoses' and len(dp['notes_radiology']) > 0:
            radiology_note += '\n\nRADIOLOGY NOTE: \n'
            for note_item in dp['notes_radiology']:
                radiology_note += note_item[0] + '\n'

        # only include lab results for the diagnosis task
        lab_events = ''
        # if the task is to decide which lab to order, then do not provide lab results in the input sequence
        if target_task == 'target_diagnoses' and len(dp['_labevents_verbalized']) > 0:
            lab_events += '\n\nLAB EVENTS: \n'
            for lab_item in dp['_labevents_verbalized']:
                lab_events += lab_item + '\n'

        # For GPT models, the full input will be generated dynamically when provide the input
        evaldata[dp_i]['input_raw'] = [
            patient_info,
            discharge_note,
            radiology_note,
            lab_events
        ]
         
        if 'gpt' not in llm_name.lower():
            # Truncate the input message to fit the model input length
            # For open-source models, we can cut the input directly here
            user_message_ideal = f"{discharge_note}{radiology_note}{lab_events}"
            user_message_ideal_tokens = tokenizer(user_message_ideal)['input_ids'][1:]
            patient_info_tokenized = tokenizer(patient_info)['input_ids'][1:]
            cut_rate = (left_token_count - len(patient_info_tokenized)) / len(user_message_ideal_tokens)
            if cut_rate > 1: cut_rate = 1
            cut_rates.append(cut_rate)
            
            if cut_rate < 1:
                if len(lab_events) > 0:
                    lab_events_tokenized = tokenizer(lab_events)['input_ids'][1:]
                    # make sure there are at least some lab results presented even the cut rate is very small
                    len_lab_events = max(token_count_lab, int(len(lab_events_tokenized) * cut_rate))
                else:
                    lab_events_tokenized = []
                    len_lab_events = 0
                discharge_note_tokenized = tokenizer(discharge_note)['input_ids'][1:]
                radiology_note_tokenized = tokenizer(radiology_note)['input_ids'][1:]
                cut_rate_note = (left_token_count - len_lab_events) / (len(discharge_note_tokenized) + len(radiology_note_tokenized))
                len_discharge_note = int(len(discharge_note_tokenized) * cut_rate_note)
                len_radiology_note = int(len(radiology_note_tokenized) * cut_rate_note)

                # truncated segments, starting from 1 to skip the <s> starting token
                discharge_note_cut = tokenizer.decode(discharge_note_tokenized[:len_discharge_note])
                radiology_note_cut = tokenizer.decode(radiology_note_tokenized[:len_radiology_note])
                lab_events_cut = tokenizer.decode(lab_events_tokenized[:len_lab_events])

                user_message_cut = f"{patient_info}{discharge_note_cut}{radiology_note_cut}{lab_events_cut}"

                if cut_rate_note < 1:
                    cut_count_user_msg_note += 1
                else:
                    cut_count_user_msg_end += 1
            else:
                user_message_cut = f"{patient_info}{user_message_ideal}"
            
            # Option 1: use our function to fill in prompt template -> not up to date
            # evaldata[dp_i]['input'] = model_specific_prompt(prompt_system_role, prompt_task_instruction, user_message_cut, prompt_task_instruction_end)
            # Option 2: use huggingface chat template
            if 'mistral' in llm_name.lower():
                messages_this = [
                    {"role": "user", "content": prompt_system_role + '\n' + prompt_task_instruction + '\n\n' + user_message_cut + '\n' + prompt_task_instruction_end}
                ]
            else:
                messages_this = [
                    {"role": "system", "content": prompt_system_role + '\n' + prompt_task_instruction},
                    {"role": "user", "content": user_message_cut + '\n' + prompt_task_instruction_end}
                ]
          
            if llm_name in ['mistralai/Mistral-7B-Instruct-v0.3' , 'meta-llama/Meta-Llama-3-8B-Instruct']:
                inputs_ = tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True) 
            else:
                inputs_ = user_message_cut + '\n' + prompt_task_instruction
             
            evaldata[dp_i]['input'] = inputs_ # tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True) 

            # Prepare target sequence used for seq2seq model training
            if target_task == 'target_diagnoses':
                gt_codes = [item[0] for item in dp['target_diagnoses']]
                evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
            elif target_task == 'target_procedures':
                gt_codes = [item[0] for item in dp['target_procedures']]
                evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
            elif target_task == 'target_laborders':
                # if use lab item names, but this is outdated implementation
                # gt_names = [item.split(":")[0] for item in dp['_labevents_verbalized']]
                # evaldata[dp_i]['target_gold'] = ". ".join(gt_names)
                # if use lab item names
                gt_names = [labitem_labels[str(item[0])] for item in dp['target_laborders']]
                evaldata[dp_i]['target_gold'] = "\n".join(gt_names)
                # if use lab LOINC code
                # gt_codes = [item[0] for item in dp['target_laborders']]
                # evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
            elif target_task == 'target_prescriptions':
                gt_names = [item[4] for item in dp['target_prescriptions']]
                evaldata[dp_i]['target_gold'] = "\n".join(gt_names)
            else:
                assert False, 'Not implemented for other tasks'

            gen_dp = {
                'id': dp['hadm_id'],
                'text': evaldata[dp_i]['input'] + evaldata[dp_i]['target_gold'],
                'input': evaldata[dp_i]['input'],
                'target_gold': evaldata[dp_i]['target_gold'],
                'eval_gold': evaldata[dp_i]['target_gold'],
            }
            gendata.append(gen_dp)

    if 'gpt' not in llm_name.lower():
        print(f'{cut_count_user_msg_note}/{len(evaldata)} input sequence cut discharge + radiology note')
        print(f'{cut_count_user_msg_end}/{len(evaldata)} input sequence cut the end of patient record')
        print(f'cut rates mean: {mean(cut_rates)}, medium: {median(cut_rates)}, min: {min(cut_rates)}, max: {max(cut_rates)}')
        # Statistics of input length and target_gold of the gen dataset
        lengths_i = [len(tokenizer(dp['input'])['input_ids'][1:]) for dp in gendata]
        lengths_o = [len(tokenizer(dp['target_gold'])['input_ids'][1:]) for dp in gendata]
        print('Statistics of number of tokens in input and target_gold of the gen dataset:')
        for list_this in [lengths_i, lengths_o]:
            print(f"Mean: {mean(list_this)}")
            print(f"Median: {median(list_this)}")
            print(f"Min: {min(list_this)}")
            print(f"Max: {max(list_this)}")
         
    pickle.dump(evaldata, open(evaldata_save_path, 'wb'))
 
del model 
del tokenizer     
with torch.no_grad():   
    results = copy.deepcopy(evaldata) 
    if os.path.exists(f'cli-data/{target_task}_output_est/{save_name}_first.json'):
        print('\n-> loading previous results\n')
        with open(f'cli-data/{target_task}_output_est/{save_name}_first.json', 'r') as f:
            results_2_first = json.load(f)
        with open(f'cli-data/{target_task}_output_est/{save_name}_last.json', 'r') as f:
            results_2_last = json.load(f)
        with open(f'cli-data/{target_task}_output_est/{save_name}_avg.json', 'r') as f:
            results_2_avg = json.load(f)
        with open(f'cli-data/{target_task}_output_est/{save_name}_sample_avg.json', 'r') as f:
            results_2_sample_avg = json.load(f)
        with open(f'cli-data/{target_task}_output_est/{save_name}_sum.json', 'r') as f:
            results_2_sum = json.load(f)
        ids_done = [result['hadm_id'] for result in results_2_first]
    else:
        results_2_first = []
        results_2_last = []
        results_2_avg = []
        results_2_sample_avg = []
        results_2_sum = []
        ids_done = []
    print(f'Produced output: {len(ids_done)}/{len(evaldata)}')
    # skip inference for the data instances that already have generated output
    results = [dp for dp in results if dp['hadm_id'] not in ids_done]
     
    print('\n# Data Samples', len(results), '\n')
    if not os.path.exists(f'cli-data/{target_task}_output_est/runtime'):
        os.makedirs(f'cli-data/{target_task}_output_est/runtime')
    global_verbose_flag = False  
    for batch_i in tqdm(range(len(results) // batch_size_big), desc=f"{target_task}, {llm_name}, inference"):
        input_dps = results[batch_i * batch_size_big: (batch_i + 1) * batch_size_big]
        
        t1 = time.time()
        outputs_first, outputs_last, outputs_avg, outputs_sample_avg, outputs_sum, outputs_sample_sum = inference(input_dps, verbose=global_verbose_flag)
        for in_batch_i, result in enumerate(input_dps): 
            results_2_first.append(
                {
                'hadm_id': result['hadm_id'],
                'output': outputs_first[in_batch_i]
            }
            )
            results_2_last.append(
                {
                'hadm_id': result['hadm_id'],
                'output': outputs_last[in_batch_i]
            }
            )
            results_2_avg.append(
                {
                'hadm_id': result['hadm_id'],
                'output': outputs_avg[in_batch_i]
            }
            )
            results_2_sample_avg.append(
                {
                'hadm_id': result['hadm_id'],
                'output': outputs_sample_avg[in_batch_i]
            }
            ) 
            results_2_sum.append(
                {
                'hadm_id': result['hadm_id'],
                'output': outputs_sum[in_batch_i]
            }
            )  
            global_verbose_flag = False 
        with open(os.path.join(f'cli-data/{target_task}_output_est', f"{save_name}_first.json"), 'w', encoding='utf-8') as f:
            json.dump(results_2_first, f, indent=4)
        with open(os.path.join(f'cli-data/{target_task}_output_est', f"{save_name}_last.json"), 'w', encoding='utf-8') as f:
            json.dump(results_2_last, f, indent=4)
        with open(os.path.join(f'cli-data/{target_task}_output_est', f"{save_name}_avg.json"), 'w', encoding='utf-8') as f:
            json.dump(results_2_avg, f, indent=4)
        with open(os.path.join(f'cli-data/{target_task}_output_est', f"{save_name}_sample_avg.json"), 'w', encoding='utf-8') as f:
            json.dump(results_2_sample_avg, f, indent=4)
        with open(os.path.join(f'cli-data/{target_task}_output_est', f"{save_name}_sum.json"), 'w', encoding='utf-8') as f:
            json.dump(results_2_sum, f, indent=4)
         
        t2 = time.time()
        with open(os.path.join(f'cli-data/{target_task}_output_est/runtime', f"{save_name}_runtime_{batch_i}.txt"), 'w' ) as f:
            f.write(f"{t2 - t1}")
            