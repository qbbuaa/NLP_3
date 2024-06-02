import jieba
from gensim.models import Word2Vec
import json
from collections import Counter

def get_useless(list):
    with open(list, 'r', encoding='utf-8') as file:
        useless = set([line.strip() for line in file.readlines()])
    return useless

def get_list(texts_dic,stopwords):
    sentences_list = []
    for novel,text in texts_dic.items():
            for line in text:
                words = jieba.lcut(line)
                sentences_list.append([word for word in words if word not in stopwords])
    return sentences_list

def get_words_dic_r(texts_dic,stopwords):
    texts_dic_r = {}
    for novel,text in texts_dic.items():
        text = ''.join(text)
        words = jieba.lcut(text)
        word_counts =Counter([word for word in words if word not in stopwords])
        for word,count in word_counts.items():
            if count >= 5:
                if not word in texts_dic_r:
                    texts_dic_r[word] = ['《'+novel+'》',count]
                else :
                    texts_dic_r[word][0] += '、《'+novel+'》'
                    texts_dic_r[word][1] += count
    return texts_dic_r

if __name__ == '__main__':

    with open('texts_dic.json', 'r') as json_file:
        texts_dic = json.load(json_file)

    stopwords = get_useless("cn_stopwords.txt")

    words_dic_r = get_words_dic_r(texts_dic,stopwords)

    sentences_list = get_list(texts_dic,stopwords)

    with open('texts_dic_r.json', 'w') as json_file:
        json.dump(words_dic_r, json_file)

    with open('sentences_list.json', 'w') as json_file:
        json.dump(sentences_list, json_file)

    print('training')
    model = Word2Vec(sentences=sentences_list, hs=1, min_count=10, window=5,
                     vector_size=200,sg=0, workers=16,epochs=100)
    model.save('model.model')

