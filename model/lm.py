import torch
import torch.nn as nn
import torch.nn.functional as F


def top5(output):
    device = output.device
    output = output.topk(5)
    value = output[0].softmax(dim=-1).squeeze().cumsum(dim=-1).cpu().tolist()

    indices = output[1].squeeze().cpu().tolist()
    flag = torch.rand(1).item()
    for i in range(len(value)):
        if flag < value[i]:
            return torch.tensor([[indices[i]]]).to(device)


class languageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(languageModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embLinear = nn.Linear(embedding_dim, hidden_dim)

        # 网络主要结构
        self.lm = nn.LSTM(hidden_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 进行分类
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, inputs, labels=None):
        # print(input.shape)
        inputsEmbeds = self.embLinear(F.relu(self.dropout(self.embeddings(inputs))))
        output, hidden = self.lm(inputsEmbeds)
        output = self.linear(output.reshape(output.shape[0] * output.shape[1], -1))

        if labels is not None:
            lossFunc = nn.CrossEntropyLoss()
            loss = lossFunc(output, labels.view(-1))
            return output, loss

        return output

    def generate(self, inputs, maxLen, bos=torch.tensor([[1]]), eos=2):
        predictList = []
        probList = []
        inputsEmbeds = self.embLinear(F.relu(self.dropout(self.embeddings(inputs))))
        output, hidden = self.lm(inputsEmbeds)
        # eos 1*1
        output = self.embLinear(F.relu(self.dropout(self.embeddings(bos.to(inputs.device)))))
        # print("out", output.shape)
        # eos 1*1*emb
        hidden = (hidden[0], torch.zeros_like(hidden[1]))
        for i in range(maxLen):
            # print(i)
            # print(hidden[0].shape)
            # print(hidden[1].shape)
            output, hidden = self.lm(output, hidden)
            wordOutput = self.linear(output)
            # print("liner",wordOutput.shape)
            # wordOutput 1*1*vocab
            probList.append(wordOutput)
            # wordOutput 1*1

            wordOutput = top5(wordOutput)
            predictList.append(wordOutput.squeeze().item())
            if predictList[-1] == eos:
                break
            output = self.embLinear(F.relu(self.dropout(self.embeddings(wordOutput))))
            # print("embedding", output.shape)

        return predictList, torch.cat(probList, dim=0)
