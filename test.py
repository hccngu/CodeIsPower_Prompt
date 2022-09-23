"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""
import os
import torch
from test_api import test_api
from test_api import RobertaEmbeddings
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaModel
import numpy as np
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='SST-2', type=str, help='[SST-2, Yelp, AGNews, TREC, MRPC, SNLI]')
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--suffix", default='', type=str)
args = parser.parse_args()

task_name = args.task_name
device = 'cuda:{}'.format(args.cuda)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# pre_pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
if task_name == 'AGNews':
    pre_pre_str = "World news : Eight to Face Court on Terror Murder Plot Charges Eight men will appear in court tomorrow accused of plotting terrorist outrages in Britain and the United States . \
                  Technology news : China Mobile to set up experimental 3G network in Beijing China Mobile Communications, currently the largest operator of mobile communication services in China, will establish a 3G (third-generation) experimental network in Beijing at the beginning of next year, according to the China-based Xinhua Online . \
                  Business news : US Air Needs Cuts to Entice Investors  ALEXANDRIA, Va. (Reuters) - US Airways Group Inc. probably  will need more cost cuts or a boost in revenue to attract the  \$250 million in equity it says it needs to leave bankruptcy  next year, the company said on Thursday . \
                  Sports news : Keenan McCardell Could Start for Chargers (AP) AP - Newly acquired wide receiver Keenan McCardell will make his season debut on Sunday and might even start for the San Diego Chargers in their road game against the Carolina Panthers"
    # n_prompt_tokens = len(tokenizer.encode(pre_pre_str, add_special_tokens=False))              
elif task_name == 'TREC':
    pre_pre_str = "Entity question : Stuart Hamblen is considered to be the first singing cowboy of which medium ? \
                Human question : Who are the nomadic hunting and gathering tribe of the Kalahari Desert in Africa ? \
                Description question : What 's the proud claim to fame of the young women who formed Kappa Alpha Theta ? \
                Location question : What German city do Italians call The Monaco of Bavaria ? \
                Numeric question : What date is Richard Nixon 's birthday ? \
                Abbreviation question : What does BMW stand for"
elif task_name == 'SNLI':
    pre_pre_str = "People using an outdoor ice skating rink ? No , The people are on a plane . \
                   A little girl on a couch holding an infant ? Maybe, A girl holds her sister . \
                   An asian child in a baseball cap crying while he is being held ? Yes , A person is holding an upset child"
elif task_name == 'MRPC':
    pre_pre_str = "The 2002 second quarter results don 't include figures from our friends at Compaq ? Yes , The year-ago numbers do not include figures from Compaq Computer . \
                   The Securities and Exchange Commission has also initiated an informal probe of Coke ? No , That federal investigation is separate from an informal inquiry by the Securities and Exchange Commission"
elif task_name == 'SST-2':
    pre_pre_str = "instead of simply handling conventional material in a conventional way , secretary takes the most unexpected material and handles it in the most unexpected way . It was great . \
                   there is a certain sense of experimentation and improvisation to this film that may not always work . It was bad"
    # n_prompt_tokens = 162 
elif task_name == 'Yelp':
    pre_pre_str = "I live just down Commonwealth and absolutely going to Common Market to get one of there deli sandwiches!  The Californian is my favorite.  They have a great wine and beer selection as well and have wine tastings and other events throughout the week!  Check 'em out! . It was great . \
                   Wir warteten gestern 45 min bis jemand die Bestellung aufnahm. In der Zeit kamen auf unser Winken vier Mitarbeiter an den Tisch, die uns alle erkl\u00e4rten, dass es nicht ihr Bereich sei. Beim Bezahlen ging es auch ewig. Ganz klar gibt es am gleichen Ort L\u00e4den mit besserem Service. . It was bad"
else:
    n_prompt_tokens = 50
    print('No in-context tech.')

# middle_str = ' ? <mask> .'


for seed in [8, 13, 42, 50, 60]:  #  
    torch.manual_seed(seed)
    np.random.seed(seed)
    # best = torch.load(f'./hcc03-9_7-results/{task_name}/{seed}/best.pt').to(device).view(n_prompt_tokens, -1)

    def sentence_fn(test_data):
        """
        This func can be a little confusing.
        Since there are 2 sentences in MRPC and SNLI each sample, we use the same variable `test_data` to represent both.
        test_data is actually a <dummy_token>. It is then replaced by real data in the wrapped API.
        For other 4 tasks, test_data must be used only once, e.g. pre_str + test_data + post_str
        """
        # SST-2: '%s . %s . It was %s .'
        if task_name in ['SST-2', 'Yelp']:
            post_str = ' . It was <mask> .'
            return pre_pre_str + ' . ' + test_data + post_str
        elif task_name == 'AGNews':
            pre_str = ' . <mask> news : '
            return pre_pre_str + pre_str + test_data
        elif task_name == 'TREC':
            # %s . %s question: %s 
            pre_str = ' . <mask> question : '
            return pre_pre_str + pre_str + test_data
        elif task_name in ['SNLI', 'MRPC']:
            middle_str = ' ? <mask> , '
            return pre_pre_str + ' . ' + test_data + middle_str + test_data
        
        # return pre_str + test_data + middle_str + test_data


    def embedding_and_attention_mask_fn(embedding, attention_mask):
        # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
        # prepad = torch.zeros(size=(1, 1024), device=device)
        # pospad = torch.zeros(size=(embedding.size(1) - n_prompt_tokens - 1, 1024), device=device)
        # return embedding + torch.cat([prepad, best, pospad]), attention_mask
        return embedding, attention_mask

    predictions = torch.tensor([], device=device)
    for res, _, _ in test_api(
        sentence_fn=sentence_fn,
        embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
        # embedding_and_attention_mask_fn=None,
        test_data_path=f'./test_datasets/{task_name}/encrypted.pth',
        task_name=task_name,
        device=device
    ):
        if task_name in ['SST-2', 'Yelp']:
            c0 = res[:, tokenizer.encode("bad", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("great", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'AGNews':
            c0 = res[:, tokenizer.encode("World", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Sports", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("Business", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("Technology", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'TREC':
            c0 = res[:, tokenizer.encode("Description", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Entity", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("Abbreviation", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("Human", add_special_tokens=False)[0]]
            c4 = res[:, tokenizer.encode("Numeric", add_special_tokens=False)[0]]
            c5 = res[:, tokenizer.encode("Location", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3, c4, c5]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'SNLI':
            c0 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Maybe", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'MRPC':
            c0 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])

    save_file_name = task_name + args.suffix
    if not os.path.exists(f'./predictions/{save_file_name}'):
        os.makedirs(f'./predictions/{save_file_name}')
    with open(f'./predictions/{save_file_name}/{seed}.csv', 'w+') as f:
        wt = csv.writer(f)
        wt.writerow(['', 'pred'])
        wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())



