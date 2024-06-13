This repository contains code accompanying the paper **A Systematic Evaluation of Decoding-Free Generative Candidate Selection Methods**.

# MCQ 

For MCQ datasets, to execute the full decoding method, run `mcq_decoding.py` and specify the desired LM and dataset arguments. For example,
```bash
python mcq_decoding.py --model_name meta-llama/Meta-Llama-3-8B --dataset commonsense_qa
```

To execute estimation methods, run `mcq_estimation.py` and specify the LM, dataset. For example,
```bash
python mcq_estimation.py --model_name meta-llama/Meta-Llama-3-8B --dataset commonsense_qa
```

The scripts download and preprocess data, perform inference, and compute the corresponding metrics, which are stored in `results/date/`. In particular, `mcq_estimation.py` computes the logit once for all estimation methods.  


The variable names for these arguments are \verb|model_name|, \verb|dataset|, with their corresponding range of values as follows. 

```python
model_name: {meta-llama/Meta-Llama-3-8B, 
             meta-llama/Meta-Llama-3-8B-Instruct,
             mistralai/Mistral-7B-v0.3, 
             mistralai/Mistral-7B-Instruct-v0.3,   
             google/flan-t5-xl}
dataset: {commonsense_qa, mmlu, gpqa, big_bench, arc} 
```

# Clibench
Download the test data of the four clinical decision tasks from [CliBench]((https://drive.google.com/drive/folders/1V0UPVFD1a3ofrIpa1EgMmrlEwZgAl8gC?usp=share_link)) 
 
To execute estimation methods, run `clibench_estimation.py` and specify the LM, dataset. For example,
```bash
python clibench_estimation.py --model_name meta-llama/Meta-Llama-3-8B --target-task target_diagnoses
```