import random
from typing import List


def load_from_file(path: str) -> List[List[str]]:
    datas = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split(' ')
            datas.append(["".join(line)])
    return datas


# 函 谷 壮 皇 居

# <s> 函 谷 壮 皇 居  </s>
#  0  5 7 9 8   22 1
def encode(strData, word2id):
    inputIdx = [word2id.get('<s>')] + [word2id.get(j, word2id['unk']) for j in strData] + [word2id.get('</s>')]
    return inputIdx


def decode(idxData, id2word):
    special = [1, 2, 0]
    inputs = [id2word.get(j, 'unk') if j not in special else '' for j in idxData]
    return "".join(inputs)


def getData(dataPathIn='../resources/couplet/test/in.txt', dataPathOut='../resources/couplet/test/out.txt',
            word2id=None, id2word=None):
    dataIn = load_from_file(dataPathIn)
    dataOut = load_from_file(dataPathOut)

    assert len(dataIn) == len(dataOut)

    dataInput = []
    dataLabel = []
    for i in range(len(dataIn)):
        # if "poem" in dataPathIn:
        # seq = random.randint(0, 5)
        if 'poem' in dataPathIn:
            input = dataIn[i][0]  # 秦 川 雄 帝 宅
        else:
            input = dataIn[i][0][0] + dataOut[i][0][0] # 春天

        if "poem" in dataPathIn:
            label = dataOut[i][0]  # 函 谷 壮 皇 居
        else:
            label = dataIn[i][0] + '。' + dataOut[i][0] # 整个对联
        dataInput.append(input)
        dataLabel.append(label)

    # 多少语料小童
    assert len(dataInput) == len(dataLabel)

    dataInputIdx = []  # [0  5 7 9 8   22 1],[]
    dataLabelIdx = []

    # word index
    for i in range(len(dataInput)):
        inputIdx = encode(dataInput[i], word2id)
        labelIdx = encode(dataLabel[i], word2id)
        dataInputIdx.append(inputIdx)
        dataLabelIdx.append(labelIdx)
    return [dataInputIdx, dataLabelIdx]
