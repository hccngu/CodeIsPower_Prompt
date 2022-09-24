from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from fsc_reward import PromptedClassificationReward


class PromptedClassificationDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        class_labels: List[str]
    ):
        assert len(source_texts) == len(class_labels)
        self.source_texts = source_texts
        self.class_labels = class_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'class_labels': self.class_labels[idx]}
        return item


def make_few_shot_classification_dataset(
        config: "DictConfig") -> Tuple[PromptedClassificationDataset]: 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, class_labels, num_classes, verbalizers, template = \
            load_few_shot_classification_dataset(config.dataset, 
                                                 config.dataset_seed, 
                                                 split, config.base_path, 
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                    class_labels)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'],
            num_classes, verbalizers, template)


def make_contest_dataset(
        config: "DictConfig") -> Tuple[PromptedClassificationDataset]:
    data_dict = {}
    for split in ['train', 'dev']:
        source_texts, class_labels, num_classes, verbalizers, template = \
            load_contest_dataset(config.dataset,
                                                 config.dataset_seed,
                                                 split, config.base_path,
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts,
                                                    class_labels)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], num_classes, verbalizers, template)


def load_few_shot_classification_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int
) -> Tuple[List[str]]:
    assert dataset in ['agnews', 'cr', 'mr', 'sst-2', 
                       'sst-5', 'yelp-2', 'yelp-5']
    ####################改
    assert split in ['train', 'dev', 'test']
    assert num_shots in [16]

    seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
    ####################改
    seed_path = seed_dict[dataset_seed]
    filepath = f'{num_shots}-shot/{dataset}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    if 'text' in df:
        source_texts = df.text.tolist()
    else: 
        source_texts = df.sentence.tolist()
    class_labels = df.label.tolist()

    verbalizers = get_dataset_verbalizers(dataset)
    num_classes = len(verbalizers)
    print(df)

    template = None
    if dataset == 'agnews': 
        template = "<mask> {prompt} {sentence_1}"

    return (source_texts, class_labels, 
            num_classes, verbalizers, template)


def load_contest_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int
) -> Tuple[List[str]]:
    assert dataset in ['AGNews', 'MRPC', 'SNLI', 'SST-2',
                       'TREC', 'Yelp']
    ####################改
    assert split in ['train', 'dev']

    seed_dict = {0:'8', 1:'13', 2:'42', 3:'50', 4:'60'}
    ####################改
    seed_path = seed_dict[dataset_seed]
    filepath = f'contest/{dataset}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    source_texts, class_labels = None, None
    if dataset in ['SNLI', 'MRPC']:
        source_texts = df.iloc[:, 0:2].tolist()
        class_labels = df.iloc[:, 2].tolist()
    else:
        source_texts = df.iloc[:, 0].tolist()
        class_labels = df.iloc[:, 1].tolist()

    verbalizers, verb_label_dicts = get_contest_verbalizers(dataset)
    num_classes = len(verbalizers)

    class_labels = encode_contest_label(class_labels, verbalizers, verb_label_dicts)

    template = None
    if dataset in ['AGNews', 'TREC']:
        template = "<mask> {prompt} {sentence_1}"
    elif dataset in ['SST-2', 'Yelp']:
        template = "{prompt} <mask> {sentence_1}"
    elif dataset in ['SNLI', 'MRPC']:
        template = "{sentence_1} {prompt} <mask> {sentence_2}"

    return (source_texts, class_labels,
            num_classes, verbalizers, template)


def encode_contest_label(labels: list, verbalizers: list, verb_label_dict: dict):
    encoded_labels = []
    for label in labels:
        encode_num = verbalizers.index(verb_label_dict[label])
        encoded_labels.append(encode_num)
    return encoded_labels


def get_dataset_verbalizers(dataset: str) -> List[str]: 
    if dataset in ['sst-2', 'yelp-2', 'mr', 'cr']:
        verbalizers = ['\u0120terrible', '\u0120great'] # num_classes
    elif dataset == 'agnews': 
        verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
    elif dataset in ['sst-5', 'yelp-5']:
        verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', 
                       '\u0120good', '\u0120great'] # num_classes
    return verbalizers


def get_contest_verbalizers(dataset: str):
    verbalizers, verb_label_dict = None, None
    if dataset in ['SST-2', 'Yelp']:
        verbalizers = ['bad', 'great']
        verb_label_dict = {'negative': 'bad', 'positive': 'great'}
    elif dataset == 'AGNews':
        verbalizers = ['World', 'Sports', 'Business', 'Technology']
        verb_label_dict = {'world': 'World', 'sports': 'Sports', 'business': 'Business', 'technology': 'Technology'}
    elif dataset == 'MRPC':
        verbalizers = ['Yes', 'No']
        verb_label_dict = {'Equivalent': 'Yes', 'NotEquivalent': 'No'}
    elif dataset == 'SNLI':
        verbalizers = ['No', 'Maybe', 'Yes']
        verb_label_dict = {'Contradiction': 'No', 'Neutral': 'Maybe', 'Entailment': 'Yes'}
    elif dataset == 'TREC':
        verbalizers = ['Human', 'Description', 'Numeric', 'Entity', 'Location', 'Abbreviation']
        verb_label_dict = {'human': 'Human', 'description': 'Description', 'numeric': 'Numeric', 'entity': 'Entity', 'location': 'Location', 'abbreviation': 'Abbreviation'}
    return verbalizers, verb_label_dict


@dataclass
class FewShotClassificationDatasetConfig:
    dataset: str = "???"
    dataset_seed: Optional[int] = None 
    base_path: str = './data'
    num_shots: int = 16


def make_prompted_classification_reward(
    num_classes: int,
    verbalizers: List[str],
    template: Optional[str],  
    config: "DictConfig") -> PromptedClassificationReward:
    return PromptedClassificationReward(config.task_lm, config.is_mask_lm, 
                                        config.compute_zscore, 
                                        config.incorrect_coeff, 
                                        config.correct_coeff,
                                        num_classes, verbalizers, template)


@dataclass
class PromptedClassificationRewardConfig:
    task_lm: str = 'distilroberta-base'
    is_mask_lm: Optional[bool] = None
    compute_zscore: bool = True
    incorrect_coeff: float = 180.0
    correct_coeff: float = 200.0
