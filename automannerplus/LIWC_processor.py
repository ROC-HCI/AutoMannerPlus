"""
Created on Tue Jun 28 3:19:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""

def match(LIWCDic, word):
    if word in LIWCDic:
        return LIWCDic[word]
    
    for i in range(1,len(word)):
        key = word[:i] + "*"
        if key in LIWCDic:
            return LIWCDic[key]
    return list()



def ReadLIWCDictionary(path):

    f = open(path)
    lines = f.readlines()
    f.close()
    
    dic = {}

    for line in lines:
        parts = line.lstrip().rstrip().split("\t")

        values = list()
        for i in range(1, len(parts)):
#            print(parts[0], parts[i])
            values.append(int(parts[i]))

        dic[parts[0]] = values

    return dic

###########################################################################
def ReadLIWCCategories(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    categories = lines[0].split("\r")
    catdic = {}
    
    for cat in categories:
        catparts = cat.split("\t")
        catdic[int(catparts[0])] = catparts[1]
    return catdic

###########################################################################

def main():
    LIWCDic = ReadLIWCDictionary('./liwcdic2007.dic')
    categories = ReadLIWCCategories('./liwccat2007.txt')

    word = "a"

    print(match(LIWCDic, word))
###########################################################################
if __name__ == "__main__":
    main()
