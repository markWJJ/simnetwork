import numpy as np
import pickle
import jieba
from gensim.models import Word2Vec



class BuildVocab(object):

    def __init__(self):
        self.token2id={}
        self.id2token={}
        self.index=len(self.token2id)
        self.emb_mat=[]
        self.extra_symbol=[]


    def deal_train_dev(self,*args,word2vec_path):
        '''
        word2vec模型中的词在后
        :param args: train/dev/test path
        :return:
        '''
        pretrain_token = {}
        pretrain_emb=[]
        pretrain_token_index=0
        untoken = {'<UNK>':0,'<PAD>':1}
        untoken_index=2
        if word2vec_path is not None:
            self.w2v_model = Word2Vec.load(word2vec_path)
        else:
            self.w2v_model = {}

        for path in args:
            with open(path,'r') as f:
                for line in f:
                    lines=line.split('\t')
                    for e in lines[0].split(' '):
                       if e in self.w2v_model and e not in pretrain_token:
                            pretrain_token[e]=pretrain_token_index
                            pretrain_emb.append(self.w2v_model[e])
                            pretrain_token_index+=1
                       elif e not in self.w2v_model and e not in untoken:
                            untoken[e]=untoken_index
                            untoken_index+=1
        untoken_all_len=len(untoken)
        self.extra_symbol=list(range(0,untoken_all_len,1))

        for k,v in pretrain_token.items():
            untoken[k]=int(v)+untoken_all_len

        self.token2id=untoken

        with open('./vocab.txt','w') as fw:
            for k,v in self.token2id.items():
                fw.write(k+'\t'+str(v)+'\n')
                self.id2token[v]=k

        self.emb_mat=np.array(pretrain_emb)

        data={}
        data["token2id"]=self.token2id
        data["id2token"]=self.id2token
        data["embedding_matrix"]=self.emb_mat
        data["extra_symbol"]=self.extra_symbol
        pickle.dump(data,open("./vocab.pkl","wb"))





if __name__ == '__main__':

    bv=BuildVocab()
    bv.deal_train_dev('./WikiQA-dev.txt','./WikiQA-test.txt',word2vec_path='./word2vec.model')
    # bv.get_word2vec_model('./word2vec.model')



# for num,ele in enumerate(sentence):
#     token2id[ele]=num
#     id2token[num]=ele
# token2id["<UNK>"]=len(token2id)+1
# token2id["<PAD>"]=len(token2id)+1
#
# id2token[len(token2id)-1]="<UNK>"
# id2token[len(token2id)]="<PAD>"
#
# emb_mat=np.random.rand(len(token2id),50)
#
# data={}
# data["token2id"]=token2id
# data["id2token"]=id2token
# data["embedding_matrix"]=emb_mat
# data["extra_symbol"]=[]
#
# pickle.dump(data,open("./vocab.pkl","wb"))

