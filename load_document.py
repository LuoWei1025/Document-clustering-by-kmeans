import os

def getFilePath(filepath = './context_file/'):
    """
    获取文件夹中的所有文件路径
    """
    filenames = os.listdir(filepath)
    filenames.sort(key = lambda x:int(x[7:-4])) #对文件进行排序
    for filename in filenames:
        yield os.path.join(filepath,filename)

def loadDocument():
    """
    将文档内容加载到列表中去
    中英文文档列表对应索引加1就是对应中英文文件中的数字编号
    """
    elist,clist = [],[]
    for filepath in getFilePath():
        with open(filepath,'r',encoding='utf-8') as f:
            if filepath[15] == 'C':
                clist.append(f.readline())
            else:
                elist.append(f.readline())
    return elist,clist

if __name__ == "__main__":
    elist,clist = loadDocument()
    #print(elist[15])
    print(elist[231])
    #print(elist[11])