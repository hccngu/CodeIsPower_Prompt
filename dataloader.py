import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer
from tfidf import generate_incontext_use_tfidf

# STOP_WORDS_PATH = 'stop_words.csv'

def prepad_prompt(instruction, n_prompt_tokens, tokenizer):
    ins_id = tokenizer.encode(instruction, add_special_tokens=False)
    print("the length of instruction + in-context is " + str(len(ins_id)))
    if len(ins_id) < 50:
        ran_id = list(range(1000, 1000 + n_prompt_tokens - len(ins_id)))
        prompt = tokenizer.decode(ran_id + ins_id)
    else:
        ins_id = ins_id[:50]
        prompt = tokenizer.decode(ins_id)
    return prompt


# 从huggingface datasets脚本中读取数据
def load_hf_dataset(data_dir: str = 'datasets', task_name: str = 'SST-2', seed: int = 42, split: str = 'train') -> datasets.Dataset:
    """
    Please choose from:
    :param task_name: 'AGNews', 'MRPC', 'SNLI', 'SST-2', 'TREC', 'Yelp'
    :param seed: 8, 13, 42, 50, 60
    :param split: 'train', 'dev'
    """
    dataset = datasets.load_dataset(
        path=f'./{data_dir}/{task_name}/{task_name}.py',
        split=f'{split}_{seed}'
    )
    return dataset


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings


class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('transformer_model/roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt:', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            instruction = '. Your task is to classify the movie review as "bad" or "great" based on its content.'      
            prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
            print('prompt:', prompt)
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'sst2', split=split)
        dataset = load_hf_dataset(task_name='SST-2', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = '. Your task is to classify the review as "bad" or "great" based on its content.'      
                prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['text'].replace("\\n", " "), self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['text'].replace("\\n", " "), self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('yelp_polarity', 'plain_text', split=split)
        dataset = load_hf_dataset(task_name='Yelp', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Technology"
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                # instruction = 'Your task is to classify the news as world or sports or business or technology based on its content . '      
                instruction = "World news : Party of Brazil's President Party Stronger In its first electoral test since taking power 21 months ago, the party of Brazil's left-leaning president emerged stronger from nationwide municipal elections but could not come in first in the country's biggest and most important city, Sao Paulo . Technology news : Microsoft Warns Asian Governments of Linux Suits Microsoft Corp. (MSFT.O: Quote, Profile, Research) warned Asian governments on Thursday they could face patent lawsuits for using the Linux operating system instead of its Windows software . Business news : US Sues Sears, Accusing It of Racial Bias The Equal Employment Opportunity Commission has sued Sears, Roebuck, contending that it illegally fired an automotive repair store manager because he was black . Sports news : Keenan McCardell Could Start for Chargers (AP) AP - Newly acquired wide receiver Keenan McCardell will make his season debut on Sunday and might even start for the San Diego Chargers in their road game against the Carolina Panthers . "
                if self.args.use_tfidf:
                    print('use tfidf !!!')
                    train_data_path = './datasets/{}/{}/train.tsv'.format(self.args.task_name, self.args.seed)
                    in_contexts = generate_incontext_use_tfidf(train_data_path)
                else:
                    in_contexts = 'Technology News : Election apology starts net feud A website that apologises to the world for the US election results has been hugely successful. \
                              Sports News : NBA Today . The Suns (18-3) have the NBA\'s best record and have won nine of their last 10 games. '
                prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            if self.args.use_rlprompt:
                if self.args.seed == 8:
                    example['input_text'] = '%s . %s ResearchMemberEmployCouncilOrgan %s' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 13:
                    example['input_text'] = '%s . %s ReviewMonitorDesignReportReport %s' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 42:
                    example['input_text'] = '%s . %s ReviewOfficialChoiceLatestNews %s' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 50:
                    example['input_text'] = '%s . %s ReviewPanelScopeCategorySort %s' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 60:
                    example['input_text'] = '%s . %s GearInformationSocialResponseResources %s' % (prompt, self.tokenizer.mask_token, example['text'])
                else:
                    raise NotImplementedError
                example['target_text'] = self.label2text[example['labels']]
            else:
                example['input_text'] = '%s . %s News: %s' % (prompt, self.tokenizer.mask_token, example['text'])
                example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s News: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('ag_news', 'default', split=split)
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='AGNews', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two news as No or Yes based on their content . '      
                if self.args.use_tfidf:
                    print('use tfidf !!!')
                    train_data_path = './datasets/{}/{}/train.tsv'.format(self.args.task_name, self.args.seed)
                    in_contexts = generate_incontext_use_tfidf(train_data_path)
                else:
                    in_contexts = 'For example , None of Deans opponents picked him as someone to party with , nor was Dean asked that question . ? Yes , None of Dean \'s opponents picked him as someone to party with and Dean was not asked the question . \
                              I loved the Brazilian music I played . ? No , " I \'ve played Brazilian music , but I \'m not Brazilian . '
                prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'mrpc', split=split)
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='MRPC', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes, Maybe, No based on their content.'
                if self.args.use_tfidf:
                    print('use tfidf !!!')
                    train_data_path = './datasets/{}/{}/train.tsv'.format(self.args.task_name, self.args.seed)
                    in_contexts = generate_incontext_use_tfidf(train_data_path)
                else:
                    in_contexts = 'For example , Girl in plaid shirt riding a unicycle. ? Yes , A girl is riding. a woman sits on the rock. ? No , A woman is riding her bicycle. Bunch of people celebrating a holiday. ? Maybe , There is a cake with candles. '
                prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt: ', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            if self.args.use_rlprompt:
                if self.args.seed == 8:
                    example['input_text'] = '%s . %s UpgradeVariableServiceStackStatus %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                elif self.args.seed == 13:
                    example['input_text'] = '%s . %s MemoryListenerJSONJSONJSON %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                elif self.args.seed == 42:
                    example['input_text'] = '%s . %s ServerFolderResourceByIdHardware %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                elif self.args.seed == 50:
                    example['input_text'] = '%s . %s DirectoryDirectoryDirectoryDirectoryDirectory %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                elif self.args.seed == 60:
                    example['input_text'] = '%s . %s FramesContextValueConnectionArgs %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                else:
                    raise NotImplementedError
                example['target_text'] = self.label2text[example['labels']]
            else:
                example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token ,example['text2'])
                example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='SNLI', split=split, seed=seed)
        # dataset = datasets.load_dataset('snli', split=split)
        dataset = dataset.filter(lambda example: example['labels'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class TRECLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Description",
            1: "Entity",
            2: "Abbreviation",
            3: "Human",
            4: "Numeric",
            5: "Location"
        }
        self.args = args

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to classify the questions as description or entity or abbreviation or human or numeric or location based on its content . '      
                if self.args.use_tfidf:
                    print('use tfidf !!!')
                    train_data_path = './datasets/{}/{}/train.tsv'.format(self.args.task_name, self.args.seed)
                    in_contexts = generate_incontext_use_tfidf(train_data_path)
                else:
                    in_contexts = 'Entity question : What is a fear of bees ? Numeric question : What is Dick Clark \'s birthday ? Abbreviation question : What does BUD stand for ? '
                prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # prompt = "Entity question : Stuart Hamblen is considered to be the first singing cowboy of which medium ? \
            #         Human question : Who are the nomadic hunting and gathering tribe of the Kalahari Desert in Africa ? \
            #         Description question : What 's the proud claim to fame of the young women who formed Kappa Alpha Theta ? \
            #         Location question : What German city do Italians call The Monaco of Bavaria ? \
            #         Numeric question : What date is Richard Nixon 's birthday ? \
            #         Abbreviation question : What does BMW stand for ? "
            # prompt = "Entity question : What 's the best way to lose the flab under your chin and around your face ? Human question : What Russian composer 's Prelude in C Sharp Minor brought him fame and fortune ? Description question : How does Zatanna perform her magic in DC comics ? Location question : What U.S. state includes the San Juan Islands ? Numeric question : How many colonies were involved in the American Revolution ? Abbreviation question : What does HIV stand for ? "
            if self.args.use_rlprompt:
                if self.args.seed == 8:
                    example['input_text'] = '%s . %s DefenseMaterialInfoMovieProject %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 13:
                    example['input_text'] = '%s . %s ResultEventBrainQueryBattery %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 42:
                    example['input_text'] = '%s . %s HelperRoamingAdapterGridMsg %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 50:
                    example['input_text'] = '%s . %s DriverIntegerBenchComputerHandler %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 60:
                    example['input_text'] = '%s . %s DistanceEventArgsWriterNode %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                else:
                    raise NotImplementedError
                example['target_text'] = self.label2text[example['labels']]
            else:
                example['input_text'] = '%s . %s question : %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . Topic : %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='TREC', split=split, seed=seed)
        dataset = dataset.filter(lambda example: example['labels'] in [0, 1, 2, 3, 4, 5])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle