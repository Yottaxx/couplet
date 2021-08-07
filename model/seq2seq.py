import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lm import top5


class Seq2SeqBaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(Seq2SeqBaseModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embLinear = nn.Linear(embedding_dim, hidden_dim)

        # 网络主要结构
        self.encoder = nn.LSTM(hidden_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 进行分类
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    # input <s> 秦 川 雄 帝 宅 </s> ->  [0 5 7 9 8 1]
    # input [<s> 秦 川 雄 帝 宅 </s>,<s> 绮 殿 千 寻 起</s>] ->  [[0 5 7 9 8 1],[ 0 11 35 127 89 99 1 ]

    # decoder [ <s> 函 谷 壮 皇 居,  <s> x x x x  ]  -> [[0 5 7 9 8],[ 0 11 35 127 89 99 1 ]
    # labels  [ 函 谷 壮 皇 居 </s> ,  <s> x x x x  ]  -> [[0 5 7 9 8],[ 0 11 35 127 89 99 1 ]
    def forward(self, inputs, decoder, labels=None):
        # print(input.shape)

        # input [0 5 7 9 8 1] ->
        # 0-> [random] * 128
        # 5 -> [random] * 128
        #
        inputsEmbeds = self.embLinear(F.relu(self.dropout(self.embeddings(inputs))))


        #  <s> 函 谷 壮 皇 居 -> [0 5 7 9 8] -> [ [random]* 128 ]
        decoderEmbeds = self.embLinear(F.relu(self.dropout(self.embeddings(decoder))))

        output, hidden = self.encoder(inputsEmbeds)

        # hidden -> 当作c 语义表示向量
        #  函 谷 壮 皇 居 </s>
        # 你 好 北 理 工 </s>
        #
        output, hidden = self.decoder(decoderEmbeds, hidden)
        output = self.linear(output.reshape(output.shape[0] * output.shape[1], -1))

        # <s>  函 0.3 你 0.5 其他-> 0.2
        # 函 1.0
        # 0.3 -> 1.0
        if labels is not None:
            lossFunc = nn.CrossEntropyLoss()
            loss = lossFunc(output, labels.view(-1))
            return output, loss

        return output

    # input <s> 秦 川 雄 帝 宅 </s> ->  [0 5 7 9 8 1]
    # input [<s> 秦 川 雄 帝 宅 </s>,<s> 绮 殿 千 寻 起</s>]

    # decoder [ <s> 你 好  你 你 你 ]  -> [[0 5 7 9 8],[ 0 11 35 127 89 99 1 ]
    # <s>->你

    # bos  = <s> -> 1
    # eos = </s> ->2
    def generate(self, inputs, maxLen=64, bos=torch.tensor([[1]]), eos=2, begin_inputs=None):
        predictList = []
        probList = []

        inputsEmbeds = self.embLinear(F.relu(self.dropout(self.embeddings(inputs))))
        output, hidden = self.encoder(inputsEmbeds)

        # c => hidden
        # eos 1*1

        # <s>
        output = self.embLinear(F.relu(self.dropout(self.embeddings(bos.to(inputs.device)))))
        # print("out", output.shape)
        # eos 1*1*emb

        for i in range(maxLen):
            # print(i)
            # print(hidden[0].shape)
            # print(hidden[1].shape)

            # 你 北
            output, hidden = self.decoder(output, hidden)

            # 你 好 北 京。。。。 [0.1,0.3  0.4  .. 0.2]
            wordOutput = self.linear(output)
            # print("liner",wordOutput.shape)
            # wordOutput 1*1*vocab
            probList.append(wordOutput)
            # wordOutput 1*1
            # wordOutput = wordOutput.argmax(dim=-1)
            if len(predictList) == 0 and begin_inputs is not None:
                predictList.append(begin_inputs.squeeze().item())
                wordOutput = begin_inputs
            else:
                wordOutput = top5(wordOutput)
                predictList.append(wordOutput.squeeze().item())
            if predictList[-1] == eos: #</s>
                break
            output = self.embLinear(F.relu(self.dropout(self.embeddings(wordOutput))))
            # print("embedding", output.shape)

        return predictList, torch.cat(probList, dim=0)
