from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from load_document import loadDocument
import numpy as np

def getFreqMatrix(elist,clist):
    """
    elist,clist：英，中文档列表，列表中的每个元素都是一个文档的原文内容
    获取词频矩阵
    """
    vc = CountVectorizer()
    ewf,cwf = vc.fit_transform(elist),vc.fit_transform(clist)
    return ewf,cwf

def gettfidf(ewf,cwf):
    """
    ewf,cwf:英，中文档对应的词频矩阵
    获取对应的tfidf值
    其中每一行代表一个文档对应的各个term的tfidf值
    """
    transformer = TfidfTransformer()
    e_tfidf,c_tfidf = transformer.fit_transform(ewf),transformer.fit_transform(cwf)
    return e_tfidf,c_tfidf

def L2Normalization(matrix):
    """
    matrix：文档的tfidf矩阵，每一行为一个文档各个term的tfidf值
    输出为进行L2正则化后的矩阵
    """
    return matrix / np.sqrt(np.sum(matrix ** 2,axis=1,keepdims=True))

def cosineDistance(nmatrix):
    """
    nmatrix：L2正则化后的tfidf矩阵
    输出包含各个文档的余弦距离矩阵,例如
        1   2   3
    1   c1  c2  c3
    2   c2  c4  c5
    3   c3  c5  c6
    m[0,1]表示文档1和文档2之间的余弦距离
    """
    return np.matmul(nmatrix,nmatrix.T)

if __name__ == "__main__":
    elist,clist = loadDocument()
    ewf,cwf = getFreqMatrix(elist,clist)
    e_tfidf,c_tfidf = gettfidf(ewf,cwf)
    ematrix,cmatrix = np.array(e_tfidf.toarray()),np.array(c_tfidf.toarray())
    ent,cnt = L2Normalization(ematrix),L2Normalization(cmatrix)
    ecm,ccm = cosineDistance(ent),cosineDistance(cnt)
    a,b = np.where(ccm > 0.9)
    for i,j in zip(a,b):
        if i != j:
            print("CDoc {} CDoc{} CosineDistance {}".format(i,j,round(ccm[i][j],6)))

    a,b = np.where(ecm > 0.9)
    for i,j in zip(a,b):
        if i != j:
            print("EDoc {} EDoc{} CosineDistance {}".format(i,j,round(ecm[i][j],6)))

    
    
    



