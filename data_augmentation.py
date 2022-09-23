import os
import random
import argparse
import pandas as pd
import numpy as np


def get_synonyms(random_word):
    pass


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        # get_synonyms 获取某个单词的同义词列表
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words


def random_replace(sentence, vocab, n):
    words = sentence.strip().split(' ')
    for _ in range(n):
        new_word = vocab[random.randint(0, len(vocab) - 1)]
        random_idx = random.randint(0, len(words) - 1)
        words[random_idx] = new_word
    return ' '.join(words)


def random_deletion(sentence, p):
    words = sentence.strip().split(' ')
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words[0]

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return words[rand_int]

    return ' '.join(new_words)


# random exchange
def random_swap(sentence, n):
    words = sentence.strip().split()
    for _ in range(n):
        random_idx_1 = random.randint(0, len(words) - 1)
        random_idx_2 = random_idx_1
        count = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(words) - 1)
            count += 1
            if count > 3:
                break
        words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]

    return ' '.join(words)


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_insertion(sentence, vocab, n):
    words = sentence.strip().split(' ')
    for _ in range(n):
        new_word = vocab[random.randint(0, len(vocab) - 1)]
        random_idx = random.randint(0, len(words) - 1)
        words.insert(random_idx, new_word)
    return ' '.join(words)


def replace_sentence(sentence1, sentence2, train_data):
    random_idx = random.randint(0, train_data.shape[0] - 1)
    return sentence1.replace("\t", " "), train_data.loc[random_idx]['sent1'].replace("\t", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--insertion", default=True, action='store_true', help='random_insertion')
    parser.add_argument("--deletion", default=True, action='store_true', help='random_deletion')
    parser.add_argument("--replace", default=True, action='store_true', help='synonym_replacement')
    parser.add_argument("--swap", default=True, action='store_true', help='random_swap')
    parser.add_argument("--replace_sentence", default=True, action='store_true', help='replace_sentence for SNLI and MRPC')
    parser.add_argument("--task_name", default='AGNews', type=str)
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--stop_words_path", default='stop_words.csv', type=str)
    parser.add_argument("--add_word_num", default=3, type=int, help='add word number for insertion')
    parser.add_argument("--del_word_prob", default=0.2, type=float, help='del word prob for del')
    parser.add_argument("--swap_word_num", default=3, type=int, help='swap word number for swap')
    parser.add_argument("--replace_word_num", default=3, type=int, help='replace word number for replace')
    parser.add_argument("--save_dir", default='DA_datasets', type=str)
    

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # set stop words:
    stop_words_data = pd.read_csv(args.stop_words_path, header=None, names=['stop_words'])
    stop_words = stop_words_data['stop_words'].tolist()
    
    # load data
    train_file_path = os.path.join('datasets', args.task_name, str(args.seed), 'train.tsv')
    if args.task_name in ['SST-2', 'TREC', 'AGNews', 'Yelp']:
        train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['sent', 'labels'])
        origin_sent_num = train_data.shape[0]
        vocab = []
        for sentence in train_data['sent']:
            words = sentence.strip().split(' ')
            for word in words:
                if word not in stop_words and word not in vocab:
                    vocab.append(word)

        if args.insertion:
            for i in range(origin_sent_num):
                sentence, label = train_data.loc[i]
                new_sentence = random_insertion(sentence, vocab, args.add_word_num)
                train_data.loc[len(train_data)] = [new_sentence, label]

        if args.deletion:
            for i in range(origin_sent_num):
                sentence, label = train_data.loc[i]
                new_sentence = random_deletion(sentence, args.del_word_prob)
                train_data.loc[len(train_data)] = [new_sentence, label]

        if args.replace:
            for i in range(origin_sent_num):
                sentence, label = train_data.loc[i]
                new_sentence = random_replace(sentence, vocab, args.replace_word_num)
                train_data.loc[len(train_data)] = [new_sentence, label]

        if args.swap:
            for i in range(origin_sent_num):
                sentence, label = train_data.loc[i]
                new_sentence = random_swap(sentence, args.swap_word_num)
                train_data.loc[len(train_data)] = [new_sentence, label]
    
    elif args.task_name in ['MRPC', 'SNLI']:
        train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['sent1', 'sent2', 'labels'])
        origin_sent_num = train_data.shape[0]
        vocab = []
        for sentence in train_data['sent1']:
            words = sentence.strip().split(' ')
            for word in words:
                if word not in stop_words and word not in vocab:
                    vocab.append(word)
        for sentence in train_data['sent2']:
            words = sentence.strip().split(' ')
            for word in words:
                if word not in stop_words and word not in vocab:
                    vocab.append(word)
        
        if args.insertion:
            for i in range(origin_sent_num):
                sentence1, sentence2, label = train_data.loc[i]
                new_sentence1 = random_insertion(sentence1, vocab, args.add_word_num)
                new_sentence2 = random_insertion(sentence2, vocab, args.add_word_num)
                train_data.loc[len(train_data)] = [new_sentence1.replace("\t", " "), new_sentence2.replace("\t", " "), label]

        if args.deletion:
            for i in range(origin_sent_num):
                sentence1, sentence2, label = train_data.loc[i]
                new_sentence1 = random_deletion(sentence1, args.del_word_prob)
                new_sentence2 = random_deletion(sentence2, args.del_word_prob)
                train_data.loc[len(train_data)] = [new_sentence1.replace("\t", " "), new_sentence2.replace("\t", " "), label]

        if args.replace:
            for i in range(origin_sent_num):
                sentence1, sentence2, label = train_data.loc[i]
                new_sentence1 = random_replace(sentence1, vocab, args.replace_word_num)
                new_sentence2 = random_replace(sentence2, vocab, args.replace_word_num)
                train_data.loc[len(train_data)] = [new_sentence1.replace("\t", " "), new_sentence2.replace("\t", " "), label]

        if args.swap:
            for i in range(origin_sent_num):
                sentence1, sentence2, label = train_data.loc[i]
                new_sentence1 = random_swap(sentence1, args.swap_word_num)
                new_sentence2 = random_swap(sentence2, args.swap_word_num)
                train_data.loc[len(train_data)] = [new_sentence1.replace("\t", " "), new_sentence2.replace("\t", " "), label]
        
        if args.replace_sentence:
            for i in range(origin_sent_num):
                sentence1, sentence2, label = train_data.loc[i]
                if label in ['Equivalent', 'Entailment', 'Neutral']:
                    train_data.loc[len(train_data)] = [sentence1.replace("\t", " "), sentence2.replace("\t", " "), label]
                    new_sentence1, new_sentence2 = replace_sentence(sentence1, sentence2, train_data)
                    if args.task_name == 'MRPC':
                        label = 'NotEquivalent'
                    elif args.task_name == 'SNLI':
                        label = 'Contradiction'
                    train_data.loc[len(train_data)] = [new_sentence1.replace("\t", " "), new_sentence2.replace("\t", " "), label]
        
    
    save_path = os.path.join(args.save_dir, args.task_name, str(args.seed), 'train.tsv')
    train_data.to_csv(save_path, sep="\t", index=False, header=False)
    print("Finishing.")


if __name__ == '__main__':
    main()
    # train_data = pd.read_csv('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/hanchengcheng02/Black-Box-Tuning-main/DA_datasets/MRPC/8/train.tsv', sep='\t', header=None, names=['sent1', 'sent2', 'labels'])
    # for i in range(train_data.shape[0]):
    #     print(train_data['sent1'], train_data['sent2'])