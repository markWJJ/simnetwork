import numpy as np
import pickle as pkl
import codecs, json, os, sys, jieba, re
from jieba import Tokenizer
from jieba.posseg import POSTokenizer
from collections import OrderedDict, Counter

class jieba_api(object):
    def __init__(self):
        print("----------using jieba cut tool---------")

    def init_config(self, config):
        self.config = config
        self.dt = Tokenizer()

    def build_tool(self):
        dict_path = self.config.get("user_dict", None)
        if dict_path is not None:
            import codecs
            with codecs.open(dict_path, "r", "utf-8") as frobj:
                lines = frobj.read().splitlines()
                for line in lines:
                    self.dt.add_word(line, 10000, "<baidu>")

    def cut(self, text):
        words = list(self.dt.cut(text))
        # print(words, " ".join([word for word in words if len(word) >= 1]))
        return " ".join([word for word in words if len(word) >= 1])

class cut_tool_api(object):
    def __init__(self):
        print("----------using naive cut tool---------")

    def init_config(self, config):
        self.config = config

    def build_tool(self):
        pass

    def cut(self, text):
        out = []
        char_pattern = re.compile(u"[\u4e00-\u9fa5]+")
        word_list = list(jieba.cut(text))
        for word in word_list:
            char_cn = char_pattern.findall(word)
            if len(char_cn) >= 1:
                for item in word:
                    if len(item) >= 1:
                        out.append(item)
            else:
                if len(word) >= 1:
                    out.append(word)
        return " ".join(out)

def make_dic(sent_list):
    dic = OrderedDict()
    for item in sent_list:
        token_lst = item.split()
        for token in token_lst:
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1
    return dic

def read_pretrained_embedding(embedding_path, dic, vocab_path, min_freq=3):
    if sys.version_info < (3, ):
        w2v = pkl.load(open(embedding_path, "rb"))
    else:
        w2v = pkl.load(open(embedding_path, "rb"), encoding="iso-8859-1")

    word2id, id2word = OrderedDict(), OrderedDict()
    pad_unk = ["<PAD>", "<UNK>", "<S>", "</S>"]
    for index, token in enumerate(pad_unk):
        word2id[token] = index
        id2word[index] = token

    unk_token = []
    pretrained_token = []
    for token in dic:
        if token in w2v:
            if dic[token] >= min_freq:
                pretrained_token.append(token)
        else:
            if dic[token] >= min_freq:
                unk_token.append(token)

    word_id = 4
    for index, token in enumerate(unk_token):
        word2id[token] = word_id
        id2word[word_id] = token
        word_id += 1

    for index, token in enumerate(pretrained_token):
        word2id[token] = word_id
        id2word[word_id] = token
        word_id += 1

    embed_dim = w2v[list(w2v.keys())[0]].shape[0]
    word_mat = np.random.uniform(low=-0.01, high=0.01, 
                                size=(len(word2id), embed_dim)).astype(np.float32)
    for word_id in range(len(word2id)):
        token = id2word[word_id]
        if token in w2v:
            word_mat[word_id] = w2v[token]

    pkl.dump({"token2id":word2id, "id2token":id2word, 
            "embedding_matrix":word_mat,
            "extra_symbol":pad_unk+unk_token}, open(vocab_path, "wb"), protocol=2)

def random_initialize_embedding(dic, vocab_path, min_freq=3, embed_dim=300):
    word2id, id2word = OrderedDict(), OrderedDict()
    pad_unk = ["<PAD>", "<UNK>", "<S>", "</S>"]
    for index, token in enumerate(pad_unk):
        word2id[token] = index
        id2word[index] = token

    word_id = 4
    for index, token in enumerate(dic):
        if dic[token] >= min_freq:
            word2id[token] = word_id
            id2word[word_id] = token
            word_id += 1
    word_mat = np.random.uniform(low=-0.01, high=0.01, 
                                size=(len(word2id), embed_dim)).astype(np.float32)
    pkl.dump({"token2id":word2id, "id2token":id2word, 
            "embedding_matrix":word_mat,
            "extra_symbol":pad_unk}, open(vocab_path, "wb"), protocol=2)

def utt2id(utt, token2id, pad_token, start_token=None, end_token=None):
    utt2id_list = []
    if start_token:
        utt2id_list = [token2id[start_token]]
    for index, word in enumerate(utt.split()):
        utt2id_list.append(token2id.get(word, token2id["<UNK>"]))
    if end_token:
        utt2id_list.append(token2id[end_token])
    return utt2id_list

def read_data(data_path, mode, word_cut_api, data_cleaner_api, split_type="blank"):
    with codecs.open(data_path, "r", "utf-8") as frobj:
        lines = frobj.read().splitlines()
        corpus_anchor = []
        corpus_check = []
        gold_label = []
        anchor_len = []
        check_len = []
        s = 1
        for line in lines:
            if split_type == "blank":
                content = line.split()
            elif split_type == "tab":
                content = line.split("\t")
            if mode == "train" or mode == "test":
                if len(content) >= 3:
                    try:
                        sent1 = content[0]
                        sent2 = content[1] 
                        label = int(content[2])
                        if s == 1:
                            print(sent1, word_cut_api.cut(sent1))
                            s += 1
                        if label == 1 or label == 0:
                            sent1 = data_cleaner_api.clean(sent1)
                            sent2 = data_cleaner_api.clean(sent2)
                            corpus_anchor.append(word_cut_api.cut(sent1))
                            corpus_check.append(word_cut_api.cut(sent2))
                            gold_label.append(label)
                            anchor_len.append(len(sent1))
                            check_len.append(len(sent2))
                        else:
                            continue
                    except:
                        continue
            else:
                if len(content) >= 2:
                    sent1 = content[0]
                    sent2 = content[1] 
                    sent1 = data_cleaner_api.clean(sent1)
                    sent2 = data_cleaner_api.clean(sent2)
                    corpus_anchor.append(word_cut_api.cut(sent1))
                    corpus_check.append(word_cut_api.cut(sent2))
                    anchor_len.append(len(sent1))
                    check_len.append(len(sent2))
        return [corpus_anchor, corpus_check, gold_label, anchor_len, check_len]

def utt2charid(utt, token2id, max_length, char_limit):
    utt2char_list = np.zeros([max_length, char_limit])
    for i, word in enumerate(utt.split()):
        for j, char in enumerate(word):
            if char in token2id:
                utt2char_list[i,j] = token2id[char]
            else:
                utt2char_list[i,j] = token2id["<UNK>"]

    return utt2char_list

def id2utt(uttid_list, id2token):
    utt = u""
    for index, idx in enumerate(uttid_list):
        if idx == 0:
            break
        else:
            utt += id2token[idx]
    return utt

