from datasets import load_dataset
import numpy as np 

answer2int = {
    'A': 0, 
    'B': 1,
    'C': 2,
    'D': 3, 
    'E': 4, 
    'F': 5,
    'G': 7
}
 
def load_data(args): 
    if args.dataset == 'commonsense_qa':
        return load_commonsense(args)
    elif args.dataset == 'mmlu':
        return load_mmlu(args)
    elif args.dataset == 'gpqa':
        return load_gpqa(args)
    elif args.dataset == 'big_bench':
        return load_big_bench(args)
    elif args.dataset == 'arc':
        return load_arc(args)
 
def load_arc(args):  
    dataset = load_dataset('allenai/ai2_arc', 'ARC-Easy')  
    args.prompt_system_role = 'You are a grade school student.'
     
    train_dataset = dataset['train'] 
    n_total = 0
    
    prompt_lis = []
    candidate_pool_lis = []
    answer_lis = []
    n_candidate_lis = []
    types = []
    total_length = 0
    total_candidate = 0 
    for i in range(len(train_dataset)): 
        if i < 1:
            print(train_dataset[i])  
         
        choices = train_dataset[i]['choices']['text']
        if len(choices) != 4:
            continue
        total_length += sum([len(choices[j].split(' ')) for j in range(len(choices))])
        total_candidate += len(choices)
        prompt_lis.append(train_dataset[i]['question']) 
        candidate_pool_lis.append(choices )
        if type(train_dataset[i]['answerKey']) not in types:
            types.append(type(train_dataset[i]['answerKey'])) 
        if train_dataset[i]['answerKey'] not in answer2int:
            answer_lis.append(int(train_dataset[i]['answerKey'])-1)
        else:
            answer_lis.append(answer2int[train_dataset[i]['answerKey']])
        n_total += 1
        n_candidate_lis.append(len(choices)) 
    bs = min(args.batch_size,n_total)
    batch_sizes = [bs for i in range(int(len(prompt_lis)/bs))]
    batch_sizes[-1] = int(len(prompt_lis) - np.sum(batch_sizes[:-1]))
    print('\n---------ARC-----------')
    print('# Candidate', np.unique(n_candidate_lis, return_counts=True)) 
    print('Total # question', n_total)
    print('avg candidate length', total_length / total_candidate)
    return prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes
 

def load_big_bench(args): 
    '''
    selection criterion: candidate pool size, candidate length
    ''' 
    args.prompt_system_role = 'You are a logic professor solving a series of logic deduction problems.'
    
    dataset = load_dataset('lighteval/big_bench_hard', 'logical_deduction_three_objects' )  
    print(dataset.keys())
    dataset = dataset['train'] 

    n_total = len(dataset) 
    
    print('n_total', n_total)
    prompt_lis = []
    candidate_pool_lis = []
    answer_lis = []
    n_candidate_lis = [] 
    total_length = 0
    total_candidate = 0 
    # Iterate through the dataset and print some examples
    for i in range(n_total): 
         
        choices = dataset[i]['input'].split('Options:\n')[1].split('\n')
        choices = [cand[4:] for cand in choices] # omit numbering (A) (B)
         
        prompt_lis.append(dataset[i]['input'].split('Options:\n')[0])  
        candidate_pool_lis.append(choices) 
        answer_lis.append(answer2int[dataset[i]['target'][1]])
        
        total_length += sum([len(choices[j].split(' ')) for j in range(len(choices))])
        total_candidate += len(choices)
        n_candidate_lis.append(len(choices))
        if i < 1:
            print(dataset[i])  
            print(answer_lis[i])

    bs = min(args.batch_size,n_total)
    batch_sizes = [bs for i in range(int(len(prompt_lis)/bs))]
    batch_sizes[-1] = int(len(prompt_lis) - np.sum(batch_sizes[:-1]))
     
    print('\n---------BIG-Bench-----------')
    print('# Candidate', np.unique(n_candidate_lis, return_counts=True)) 
    print('Total # question', n_total)
    print('avg candidate length', total_length / total_candidate)
    return prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes
 
 
def load_gpqa(args):  
    args.prompt_system_role = "You are a distinguished academic with expertise in multiple disciplines."
     
    dataset = load_dataset('Idavidrein/gpqa', 'gpqa_main')  
    print(dataset.keys())
    dataset = dataset['train'] 

    n_total = len(dataset) 
    
    print('n_total', n_total)
    prompt_lis = []
    candidate_pool_lis = []
    answer_lis = []
    n_candidate_lis = []
    types = []
    total_length = 0
    total_candidate = 0
    # Iterate through the dataset and print some examples
    for i in range(n_total): 
        if i < 1: 
            print(dataset[i].keys())
             
        prompt_lis.append(dataset[i]['Question']) 
        choices = [dataset[i]['Correct Answer'], dataset[i]['Incorrect Answer 1'], dataset[i]['Incorrect Answer 2'], dataset[i]['Incorrect Answer 3']]
        idx = range(4)
        idx_ = np.random.permutation(idx)
        choices_ = [choices[int(j)] for j in idx_]
        candidate_pool_lis.append(choices_ )
        answer_lis.append(np.where(idx_ == 0)[0][0]) 
        
        total_length += sum([len(choices[j].split(' ')) for j in range(len(choices))])
        total_candidate += len(choices)
 
        n_candidate_lis.append(len(choices))
     
    bs = min(args.batch_size,n_total)
    batch_sizes = [bs for i in range(int(len(prompt_lis)/bs))]
    batch_sizes[-1] = int(len(prompt_lis) - np.sum(batch_sizes[:-1]))
    
    print('\n---------GPQA-----------')
    print('# Candidate', np.unique(n_candidate_lis, return_counts=True)) 
    print('Total # question', n_total)
    print('avg candidate length', total_length / total_candidate)
    return prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes
 


def load_mmlu(args): 
    args.prompt_system_role = 'You are a scholar with extensive knowledge across various disciplines.'
     
    dataset = load_dataset('cais/mmlu', 'all')  
    print(dataset.keys())
    dataset = dataset['test'] 

    n_total = len(dataset) 
    
    print('n_total', n_total)
    prompt_lis = []
    candidate_pool_lis = []
    answer_lis = []
    n_candidate_lis = []
    types = []
    total_length = 0
    total_candidate = 0
    # Iterate through the dataset and print some examples
    for i in range(n_total): 
        if i < 1:
            print(dataset[i])
             
        prompt_lis.append(dataset[i]['question']) 
        choices = dataset[i]['choices'] 
        candidate_pool_lis.append(choices )
        total_length += sum([len(choices[j].split(' ')) for j in range(len(choices))])
        total_candidate += len(choices)

        if not type(dataset[i]['answer']) in types:
            types.append( type(dataset[i]['answer']))

        answer_lis.append(dataset[i]['answer'])
        
        # print(prompt_lis[0])
        # print(candidate_pool_lis[0])
        # print(answer_lis[0]) 
        n_candidate_lis.append(len(choices))

    bs = min(args.batch_size,n_total)
    batch_sizes = [bs for i in range(int(len(prompt_lis)/bs))]
    batch_sizes[-1] = int(len(prompt_lis) - np.sum(batch_sizes[:-1]))
    print('\n---------MMLU-----------')
    print('# Candidate', np.unique(n_candidate_lis, return_counts=True)) 
    print('Total # question', n_total)
    print('avg candidate length', total_length / total_candidate)
    return prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes
 
 
def load_commonsense(args):
    args.prompt_system_role = 'You are an intelligent assistant with a vast understanding of everyday life.'
     
    # Load the CommonSenseQA dataset 
    dataset = load_dataset(args.dataset)  
     
    train_dataset = dataset['train'] 
    n_total = len(train_dataset) 
    prompt_lis = []
    candidate_pool_lis = []
    answer_lis = []
    n_candidate_lis = []
    total_length = 0
    total_candidate = 0
    # Iterate through the dataset and print some examples
    for i in range(n_total): 
        if i < 1:
            print(train_dataset[i])
        prompt_lis.append(train_dataset[i]['question']) 
        choices = train_dataset[i]['choices']['text']

        total_length += sum([len(choices[j].split(' ')) for j in range(len(choices))])
        total_candidate += len(choices)

        candidate_pool_lis.append(choices )
         
        answer_lis.append(answer2int[train_dataset[i]['answerKey']])
        # print(prompt_lis[0])
        # print(candidate_pool_lis[0])
        # print(answer_lis[0]) 
        n_candidate_lis.append(len(choices))

    bs = min(args.batch_size,n_total)
    batch_sizes = [bs for i in range(int(len(prompt_lis)/bs))]
    batch_sizes[-1] = int(len(prompt_lis) - np.sum(batch_sizes[:-1]))
     
    print('\n---------CommonsenseQA-----------')
    print('# Candidate', np.unique(n_candidate_lis, return_counts=True)) 
    print('Total # question', n_total)
    print('avg candidate length', total_length / total_candidate)
    return prompt_lis, candidate_pool_lis, answer_lis, n_candidate_lis, batch_sizes
 