import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    model_name = model.split('/')[-1]
    cache_file = f'cache/{name}_{nsamples}_{seed}_{seqlen}_{model_name}.pt'
    try:
        test_enc = torch.load(cache_file)
        return test_enc
    except:
        pass

    tokenizer = get_tokenizer(model)
    loaders = get_wikitext2(nsamples, seed, seqlen, model, tokenizer)

    directory = '/'.join(cache_file.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(loaders, cache_file)
    return loaders

