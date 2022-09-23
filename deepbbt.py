import os
import copy
import time
import math
import pickle
import random

import torch
import argparse
import numpy as np
import cma
from fastNLP import cache_results, Tester, DataSet
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
    BartConfig,
    BartTokenizer,
    T5Config,
    T5Tokenizer,
    GPT2Config,
    GPT2Tokenizer,
)
from models.deep_modeling_roberta import RobertaForMaskedLM
from models.deep_modeling_bart import BartForConditionalGeneration
from models.deep_modeling_t5 import T5ForConditionalGeneration
from models.deep_modeling_gpt2 import GPT2LMHeadModel
from models.deep_modeling_bert import BertForMaskedLM
from models.deep_modeling_cpt import CPTForMaskedLM
from utils import hinge_loss
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large',
                    # choices=['roberta-base', 'roberta-large',
                    #          'bert-base-uncased', 'bert-large-uncased',
                    #          'facebook/bart-base', 'facebook/bart-large',
                    #          't5-small', 't5-base', 't5-large', 't5-3b',
                    #          'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                    #          ], 
                    type=str)
parser.add_argument("--model_path", default='roberta-large', type=str, help='The path of hugging face models for offline mode, default=model_name')
parser.add_argument("--task_name", default='TREC', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma1", default=1, type=float)
parser.add_argument("--sigma2", default=0.2, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str)
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)

# for Data Augmentation
parser.add_argument("--data_dir", default='datasets', type=str, help="'dataset' represents origin datasets, for DA, please select 'DA_datasets'. ")


args = parser.parse_args()

# below are free hyper-params
model_name = args.model_name
if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    from dataloader_t5 import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader
    from metrics_t5 import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    from dataloader_gpt import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader
    from metrics_gpt import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric
else:
    from dataloader import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader, TRECLoader
    from metrics import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric, TRECMetric

model_path = args.model_path
task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
bound = args.bound
sigma1 = args.sigma1
sigma2 = args.sigma2
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path
save_hiddens = True
data_dir = args.data_dir

# # fixed hyper-params
# if cat_or_add == 'add':
#     init_prompt_path = None
# else:
#     init_prompt_path = './nli_base_prompt.pt'

if task_name in ['SST-2', 'Yelp', 'MRPC']:
    num_labels = 2
elif task_name in ['SNLI']:
    num_labels = 3
elif task_name in ['AGNews']:
    num_labels = 4
elif task_name in ['TREC']:
    num_labels = 5
else:
    raise ValueError

args.bbt_version = 'deepbbt'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class LMForwardAPI:
    def __init__(self, model_name='roberta-large', model_path='roberta-large', n_prompt_tokens=50, task_name='SST-2',
                 loss_type='hinge'):
        self.model_name = model_name
        self.model_path = model_path
        if model_name in ['roberta-base', 'roberta-large']:
            self.config = RobertaConfig.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework=inference_framework,
                onnx_model_path=onnx_model_path,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.config = BertConfig.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
            self.config = BartConfig.from_pretrained(model_path)
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            self.config = T5Config.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            self.config = GPT2Config.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['fnlp/cpt-large']:
            self.config = BartConfig.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = CPTForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            raise NotImplementedError

        if random_proj == 'normal':
            self.config.output_hidden_states = True

        if inference_framework == 'ort':
            self.model.roberta = None
        self.best_prefix = torch.zeros(self.config.num_hidden_layers, n_prompt_tokens, self.config.hidden_size,
                                       device=device)
        self.best = None
        self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False) for _ in
             range(self.config.num_hidden_layers)])
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['roberta-base', 'roberta-large']:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
                embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['facebook/bart-base', 'facebook/bart-large', 'fnlp/cpt-large']:
                embedding = self.model.model.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                embedding = self.model.transformer.get_input_embeddings().weight.clone().cpu()
            else:  # T5
                embedding = self.model.get_input_embeddings().weight.clone().cpu()
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear[0].parameters():
                torch.nn.init.normal_(p, 0.0, std)
            self.intermediate_stats = [(mu, std)]
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.num_call = 0
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        if task_name == 'SST-2':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'AGNews':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'Yelp':
            self.metric = YelpPMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'MRPC':
            self.metric = MRPCMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'SNLI':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        elif task_name == 'TREC':
            self.metric = TRECMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'TRECMetric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf

    def eval(self, prompt_embedding=None, layer_id=None, test_data=None):
        self.num_call += 1
        best_prefix = self.best_prefix.clone()
        if prompt_embedding is not None:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear[layer_id](prompt_embedding).reshape(-1, self.config.hidden_size)  # Az
            best_prefix[layer_id] = prompt_embedding

        self.model.set_prompt_embedding(best_prefix)

        for k, v in train_data.items():
            train_data[k] = v.to(device)
        with torch.no_grad():
            if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    decoder_input_ids=train_data['decoder_input_ids'],
                    decoder_attention_mask=train_data['decoder_attention_mask'],
                )
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                )
            else:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )
            logits = outputs['logits']
            if random_proj == 'normal' and len(self.intermediate_stats) == 1:
                # if is the first forward pass, record the range of hidden states of each layer
                print('Calculating std for random projections...')
                if self.model_name in ['facebook/bart-base', 'facebook/bart-large',
                                        't5-small', 't5-base', 't5-large', 't5-3b',
                                        'fnlp/cpt-large',
                                        ]:
                    hidden_states = outputs['encoder_hidden_states']
                else:
                    hidden_states = outputs['hidden_states']
                for i, h in enumerate(hidden_states[1:-1]):
                    if save_hiddens:
                        hid_path = './hidstates/{}'.format(self.model_name.split('/')[-1])
                        if not os.path.exists(hid_path):
                            os.makedirs(hid_path, exist_ok=True)
                        with open('{}/hidden_{}.bin'.format(hid_path, i + 1), 'wb') as f:
                            pickle.dump(h, f)
                    print('[Layer {}]'.format(i + 1))
                    hidden = h.clone().reshape(-1).detach().cpu().numpy()
                    mu_hat = np.mean(hidden)
                    std_hat = np.std(hidden)
                    max_h = np.max(hidden)
                    min_h = np.min(hidden)
                    print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                        mu_hat, std_hat, min_h, max_h))
                    # Clipping outliers
                    clip_round = 0
                    while clip_round < 5:
                        clip_round += 1
                        min_bound = mu_hat - 3 * std_hat
                        max_bound = mu_hat + 3 * std_hat
                        hidden = np.clip(hidden, min_bound, max_bound)
                        mu_hat = np.mean(hidden)
                        std_hat = np.std(hidden)
                        max_h = np.max(hidden)
                        min_h = np.min(hidden)
                        print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                            clip_round, mu_hat, std_hat, min_h, max_h))
                    # Calculating std dev for the random projection
                    mu = 0.0
                    std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
                    print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
                    for p in self.linear[i + 1].parameters():
                        torch.nn.init.normal_(p, mu, std)
                    self.intermediate_stats.append((mu, std))
                assert len(self.intermediate_stats) == self.config.num_hidden_layers
                self.model.config.output_hidden_states = None
                print('Random projections initialized.')

            loss, perf = self.calc_metric(logits, train_data['labels'])

            if perf > self.best_train_perf:
                self.best_train_perf = perf

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                            decoder_input_ids=dev_data['decoder_input_ids'],
                            decoder_attention_mask=dev_data['decoder_attention_mask'],
                        )['logits']
                    elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                        )['logits']
                    else:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                            mask_pos=dev_data['mask_pos'],
                        )['logits']

                dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    self.best = best_prefix.clone()
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
            return loss


    def calculate_CM(self, best_prompt_embedding=None):
        # self.num_call += 1
        best_prefix = self.best_prefix.clone()
        for i in range(24):
            best_prompt_embedding[i] = torch.tensor(best_prompt_embedding[i]).type(torch.float32)  # z
            best_prompt_embedding[i] = self.linear[i](best_prompt_embedding[i]).reshape(-1, self.config.hidden_size)  # Az
            best_prefix[i] = best_prompt_embedding[i]

        self.model.set_prompt_embedding(best_prefix)

        for k, v in train_data.items():
            train_data[k] = v.to(device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=train_data['input_ids'],
                attention_mask=train_data['attention_mask'],
                mask_pos=train_data['mask_pos'],
            )
            logits = outputs['logits']
            if random_proj == 'normal' and len(self.intermediate_stats) == 1:
                # if is the first forward pass, record the range of hidden states of each layer
                print('Calculating std for random projections...')
                if self.model_name[18:] in ['facebook/bart-base', 'facebook/bart-large',
                                        't5-small', 't5-base', 't5-large', 't5-3b',
                                        'fnlp/cpt-large',
                                        ]:
                    hidden_states = outputs['encoder_hidden_states']
                else:
                    hidden_states = outputs['hidden_states']
                for i, h in enumerate(hidden_states[1:-1]):
                    if save_hiddens:
                        hid_path = './hidstates/{}'.format(self.model_name.split('/')[-1])
                        if not os.path.exists(hid_path):
                            os.makedirs(hid_path, exist_ok=True)
                        with open('{}/hidden_{}.bin'.format(hid_path, i + 1), 'wb') as f:
                            pickle.dump(h, f)
                    print('[Layer {}]'.format(i + 1))
                    hidden = h.clone().reshape(-1).detach().cpu().numpy()
                    mu_hat = np.mean(hidden)
                    std_hat = np.std(hidden)
                    max_h = np.max(hidden)
                    min_h = np.min(hidden)
                    print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                        mu_hat, std_hat, min_h, max_h))
                    # Clipping outliers
                    clip_round = 0
                    while clip_round < 5:
                        clip_round += 1
                        min_bound = mu_hat - 3 * std_hat
                        max_bound = mu_hat + 3 * std_hat
                        hidden = np.clip(hidden, min_bound, max_bound)
                        mu_hat = np.mean(hidden)
                        std_hat = np.std(hidden)
                        max_h = np.max(hidden)
                        min_h = np.min(hidden)
                        print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                            clip_round, mu_hat, std_hat, min_h, max_h))
                    # Calculating std dev for the random projection
                    mu = 0.0
                    std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
                    # temp = intrinsic_dim - std_hat * std_hat
                    # mu = mu_hat / temp
                    # std = std_hat / np.sqrt(temp)
                    print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
                    for p in self.linear[i + 1].parameters():
                        torch.nn.init.normal_(p, mu, std)
                    self.intermediate_stats.append((mu, std))
                assert len(self.intermediate_stats) == self.config.num_hidden_layers
                self.model.config.output_hidden_states = None
                print('Random projections initialized.')

        print('********* Evaluated on dev set *********')
        for k, v in dev_data.items():
            dev_data[k] = v.to(device)
        with torch.no_grad():
            logits = self.model(
                input_ids=dev_data['input_ids'],
                attention_mask=dev_data['attention_mask'],
                mask_pos=dev_data['mask_pos'],
            )['logits']

        dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])


if model_name in ['roberta-base', 'roberta-large']:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
elif model_name in ['bert-base-uncased', 'bert-large-uncased', 'fnlp/cpt-large']:
    tokenizer = BertTokenizer.from_pretrained(model_path)
elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
    tokenizer = BartTokenizer.from_pretrained(model_path)
elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
else:
    raise NotImplementedError

cache_fn = f"caches/data_{model_name.replace('/', '-')}_{data_dir}_{task_name}_{n_prompt_tokens}_{seed}.pt"

DataLoader = {
    'SST-2': SST2Loader,
    'AGNews': AGNewsLoader,
    'Yelp': YelpPLoader,
    'MRPC': MRPCLoader,
    'SNLI': SNLILoader,
    'TREC': TRECLoader,
}


@cache_results(cache_fn, _refresh=True)
def get_data(task_name, tokenizer):
    splits = ['train', 'dev']
    data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens, data_dir=data_dir).my_load(splits, seed)
    return data_bundle

data_bundle = get_data(task_name=task_name, tokenizer=tokenizer)
train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')

for ds in [train_data, dev_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)

print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])

if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'decoder_input_ids': torch.tensor(train_data['decoder_input_ids'].get(list(range(len(train_data))))),
        'decoder_attention_mask': torch.tensor(train_data['decoder_attention_mask'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'decoder_input_ids': torch.tensor(dev_data['decoder_input_ids'].get(list(range(len(dev_data))))),
        'decoder_attention_mask': torch.tensor(dev_data['decoder_attention_mask'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }
else:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }

model_forward_api = LMForwardAPI(
    model_name=model_name,
    model_path=model_path,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    loss_type=loss_type,
)

cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'maxiter': budget // (popsize * model_forward_api.config.num_hidden_layers),
    'verbose': -1,
}
if bound > 0:
    cma_opts['bounds'] = [-1 * bound, 1 * bound]

sigmas = [sigma1]
for i in range(model_forward_api.config.num_hidden_layers - 1):
    sigmas.append(sigma2)
assert len(sigmas) == model_forward_api.config.num_hidden_layers
es_list = [
    cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigmas[i], inopts=cma_opts)
    for i in range(model_forward_api.config.num_hidden_layers)
]
start_time = time.time()

for _ in range(budget // (int(popsize) * model_forward_api.config.num_hidden_layers)):
    for i, es in enumerate(es_list):
        solutions = es.ask()
        fitnesses = [model_forward_api.eval(x, i) for x in solutions]
        es.tell(solutions, fitnesses)
        model_forward_api.best_prefix[i] = model_forward_api.linear[i](
            torch.tensor(es.result.xbest).type(torch.float32)).reshape(-1,
                                                                       model_forward_api.config.hidden_size)  # set best cv

end_time = time.time()
print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
if not os.path.exists(f'./results/{task_name}/{seed}'):
    os.makedirs(f'./results/{task_name}/{seed}')

torch.save(model_forward_api.best_prefix, f=f'./results/{task_name}/{seed}/best.pt')
