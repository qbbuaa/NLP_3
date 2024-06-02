import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import json
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


def save_similar(names,model,texts_dic_r):
    for name in names:
        table = []
        for result in model.wv.similar_by_word(name, topn=10):
            table.append([result[0],texts_dic_r[result[0]][0],'{:.5f}'.format(result[1])])
        df = pd.DataFrame(table,columns=['其他关键词','所属小说','余弦相似度'])
        df.index = [1,2,3,4,5,6,7,8,9,10]
        df.to_excel(texts_dic_r[name][0]+'中“'+name+'”余弦相似度表.xlsx')

def process_keywords(keywords_dic):
    keywords_dic_r = {}
    keywords_list = []
    for name,keywords in keywords_dic.items():
        keywords_list.extend(keywords)
        for keyword in keywords:
            keywords_dic_r[keyword] = name
    return keywords_list,keywords_dic_r

def save_cluster(keywords,keywords_dic_r,labels):
    novels = []
    for keyword in keywords:
        novels.append(keywords_dic_r[keyword])
    table = [keywords,novels,labels]
    df = pd.DataFrame(table)
    df.to_excel('聚类.xlsx')

def get_highfrequence(texts_dic_r,model):
    highfrequence_list = []
    for word,value in texts_dic_r.items():
        if word in model.wv and value[1] >= 50:
            highfrequence_list.append(word)
    highfrequence_vectors = np.array([model.wv[word] for word in highfrequence_list])
    print(highfrequence_vectors)
    return highfrequence_list,highfrequence_vectors
def get_sentences_vector(sentences_list,model):
    sentences_vector = []
    for sentence in sentences_list:
        sentence_in_model = [word for word in sentence if word in model.wv]
        try:
            sentence_vector = model.wv[sentence_in_model].mean(axis=0)
            sentences_vector.append(sentence_vector)
        except:
            continue
    return sentences_vector

if __name__ == '__main__':
    model = Word2Vec.load('model.model')

    with open('texts_dic_r.json', 'r') as json_file:
        texts_dic_r = json.load(json_file)
   
    selected_novels = ['倚天屠龙记', '天龙八部', '神雕侠侣', '笑傲江湖','射雕英雄传']
    names = ['张无忌','段誉','杨过','郭靖','令狐冲']
    path = '余弦相似度表.xlsx'
    save_similar(names, model, texts_dic_r)

    highfrequence_list,highfrequence_vectors = get_highfrequence(texts_dic_r,model)
    tsne = TSNE(n_components = 2,random_state=0)
    embedding = tsne.fit_transform(highfrequence_vectors)
    label = KMeans(16).fit(embedding).labels_
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bx', 'gx', 'rx', 'cx', 'mx', 'yx', 'kx', 'b>', 'g>']
    for i in range(len(label)):
        plt.plot(embedding[i][0], embedding[i][1],colors[label[i]])
    plt.title('cluster')
    plt.savefig('cluster.png')
    plt.close()

    select = ['韦小宝', '点头', '道', '如此说来', '倒', '算', '公平', '你家', '王子', '预定', '起事', '罕贴摩', '道', '这件', '大事', '王爷',
                '主', '三家', '呼应', '夹攻', '自然', '全', '王爷', '主意', '韦小宝', '道', '父王', '的的确确', '知道', '出兵', '之后', '三家',
                '呼应', '罕贴摩', '道', '一节', '请', '王爷', '不必', '担心', '王爷', '大军', '一出', '支贵', '蒙古', '精兵', '西而东', '罗刹', '国',
                '哥萨克', '精骑', '自北', '南', '两路', '夹攻', '北京', '西藏', '活佛', '藏兵', '立刻', '攻掠', '川边', '神龙教', '奇兵', '韦小宝', '一声',
                '一拍', '大腿', '说道', '神龙教', '事', '知道', '洪', '教主', '说', '听到', '神龙教', '竟', '这项', '阴谋', '关心', '震荡', '说话', '声音',
                '发颤', '罕贴摩', '见', '神色', '异', '问道', '神龙教', '事', '王爷', '小王爷', '说']
    select_in_model = [word for word in select if word in model.wv]
    sentence_vector = model.wv[select_in_model].mean(axis=0)
    with open('sentences_list.json', 'r') as json_file:
        sentences_list = json.load(json_file)

    sentences_list.remove(select)
    sentences_vector = get_sentences_vector(sentences_list,model)
    similaritys = []
    for vector in sentences_vector:
        similarity = np.dot(vector,sentence_vector) / (np.linalg.norm(vector) * np.linalg.norm(sentence_vector))
        similaritys.append(similarity)

    max_position = similaritys.index(max(similaritys))
    print(sentences_list[max_position])

