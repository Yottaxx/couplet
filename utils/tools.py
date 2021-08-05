def getVocab(path ="../resources/couplet/vocabs" ):
    wordList = []
    with open(path, "r") as fin:
        line = fin.readline()
        while line:
            wordList.append(line.split('\n')[0])
            line = fin.readline()

    word2id = {wordList[i]: i + 1 for i in range(len(wordList))}
    word2id['pad'] = 0
    word2id['unk'] = len(word2id)
    id2word = {word2id[k]: k for q, k in enumerate(word2id)}
    return word2id, id2word
