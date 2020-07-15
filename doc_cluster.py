import numpy as np
import random
from similarity import L2Normalization
from load_document import loadDocument
from similarity import getFreqMatrix,gettfidf


def randPick(k,matrix):
    """
    k：随机挑选k个中心点
    matrix：文档对应的tfidf矩阵
    返回的是对应中心点
    """
    index,docnums = [],matrix.shape[0]
    random.seed(1)
    while index == [] or len(index) < k:
        idx= random.randint(0,docnums - 1)
        if idx not in index:
            index.append(idx)
    return matrix[index]

def getCosineDistance(seeds,docmatrix):
    """
    计算中心点与各个文档间的距离
    seeds：中心点
    docmatrix：标识文档矩阵
    """
    return np.matmul(seeds,docmatrix.T)

def redivideClusters(cosmatrix,k,docmatrix):
    """
    cosmatrix：根据中心点计算出的余弦矩阵
    将文档划分到对应的簇，并返回新划分的簇和对应簇的中心点
    """
    clusters = [[] for _ in range(k)]   #k个簇
    x,y = np.where(cosmatrix == np.max(cosmatrix,axis=0))
    for i,j in zip(x,y):
        #print(cosmatrix[i][j])
        clusters[i].append(j)
    #重新计算中心点
    nseeds = []
    for cluster in clusters:
        vectors = docmatrix[cluster]
        s = (np.sum(vectors,axis=0) / len(vectors)).tolist()
        nseeds.append(s)
       
    return clusters,np.array(nseeds)

def kmeans(k,norm):
    seeds,clusters = randPick(k,norm),[] #从文档矩阵中随机挑选k个中心点
    clusterChanged = True #聚簇收敛标志位

    while clusterChanged:
        cd = getCosineDistance(seeds,norm) #计算各文档到各中心的距离
        clusters,nseeds = redivideClusters(cd,k,norm) #将各文档划分到对应的簇
        if (np.array(nseeds) == np.array(seeds)).all(): #中心不再发生变化
            clusterChanged = False
        else:
            seeds = nseeds
    return clusters,seeds

def findRepresent(clusters,seeds,docmatrix):
    """
    clusters:聚类算法得到的类
    seeds:聚类算法的最终中心点，与clusters中的簇一一对应
    docmatrix：文档矩阵
    寻找聚类形成的3个最大的类及离类中心最近的5个文档
    """
    #获取最大的三个类
    sort_cluster = sorted(clusters,key = lambda x:len(x),reverse = True)
    top_three,top_dis = sort_cluster[:3],[]
    #计算最大的三个类中文档到其中心点的距离
    for top in top_three:
        #获取对应簇的中心点
        top_cp = np.array(seeds[clusters.index(top)])
        #获取类中的文档矩阵
        topm = np.array(docmatrix[top])
        #计算类中的各文档到中心点的距离矩阵，其维度为1*M,M为类中的文档数
        top_dis.append(np.matmul(top_cp,topm.T))

    nearest_five = []
    for i,ds in enumerate(top_dis):
        #获取类中文档号和文档与对应中心点之间距离列表
        tmp = [[top_three[i][j],ds[j]] for j in range(len(ds))]
        #对tmp按余弦距离进行排序
        tmp.sort(key = lambda x:x[1],reverse = True)
        #截取其中5个最近的文档
        nearest_five.append(tmp[:5])
    
    return top_three,nearest_five

if __name__ == "__main__":
    elist,clist = loadDocument()
    ewf,cwf = getFreqMatrix(elist,clist)
    e_tfidf,c_tfidf = gettfidf(ewf,cwf)
    ematrix,cmatrix = L2Normalization(np.array(e_tfidf.toarray())),L2Normalization(np.array(c_tfidf.toarray()))
    k = 20
    ecluster,eseeds = kmeans(k,ematrix)
    ccluster,cseeds = kmeans(k,cmatrix)

    tops,top_score = findRepresent(ccluster,cseeds,cmatrix)
    for i in range(3):
        print("中文文档top {}".format(i + 1))
        print(tops[i])
        print(top_score[i])
    print('----------------------------------------')
    tops,top_score = findRepresent(ecluster,eseeds,ematrix)
    for i in range(3):
        print("英文文档top {}".format(i + 1))
        print(tops[i])
        print(top_score[i])