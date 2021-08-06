import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import nltk.translate.bleu_score as bleu
from model.seq2seq import Seq2SeqBaseModel
from utils.args import getArgs
from utils.argsLM import getArgsLM
from utils.dataset import CustomDataset
from utils.process import encode, decode, getData
from utils.tools import getVocab
import torch.nn as nn

#对联
#args = getArgs()
#古诗
args = getArgsLM()

best_bleu = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2id, id2word = getVocab(path="./resources/couplet/vocabs")

data = getData(dataPathIn='./resources/poem/train/in.txt', dataPathOut='./resources/poem/train/out.txt',
               word2id=word2id, id2word=id2word)
dataDev = getData(dataPathIn='./resources/couplet/test/in.txt', dataPathOut='./resources/couplet/test/out.txt',
                  word2id=word2id, id2word=id2word)

# data = getData(dataPathIn='./resources/couplet/test/in.txt',dataPathOut='./resources/couplet/test/out.txt',word2id=word2id,id2word=id2word)
trainDataset = CustomDataset(data=data, word2id=word2id, id2word=id2word, device=device)
devDataset = CustomDataset(data=dataDev, word2id=word2id, id2word=id2word, device=device)

dataloader = DataLoader(trainDataset, batch_size=args.train_batch_size, shuffle=True,
                        collate_fn=trainDataset.collate_fn)

dataloaderDev = DataLoader(devDataset, batch_size=args.eval_batch_size, shuffle=False,
                           collate_fn=trainDataset.collate_fn)

model = Seq2SeqBaseModel(vocab_size=len(word2id), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, len(dataloader), args.num_train_epochs * len(dataloader))


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    # config = Config("resources/couplet/chars_sort.txt")
    global best_bleu
    print("start training...")
    for epoch in range(args.num_train_epochs):
        model.train()  # set mode to train

        losses = AverageMeter()
        clips = AverageMeter()

        optimizer.zero_grad()
        tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)

        for data in tk:
            inputs, decoderInput, labels = data
            logits, loss = model(inputs, decoderInput, labels)

            losses.update(loss.item(), logits.size(0))

            loss.backward()
            clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clips.update(clip.item(), logits.size(0))
            tk.set_postfix(loss=losses.avg,clips = clips.avg)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        bleu_score =  eval()
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(model.state_dict(),
                       "bsz"+str(args.train_batch_size)+"ed"+str(args.embedding_dim)+"tb"+str(args.train_batch_size)+"hd"+str(args.hidden_dim)+"bs"+str(args.num_layers)+"lr"+str(args.learning_rate)+'seq2seqPoem{}.pt'.format(best_bleu))

        generate("你好")
        generate("小王")


def generate(inputs):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(encode(inputs, word2id)).unsqueeze(dim=0).to(device)
        predictList, prbList = model.generate(inputs, maxLen=64)
        print(predictList)
        print(decode(predictList, id2word))


def eval():
    model.eval()
    bleu = AverageMeter()

    optimizer.zero_grad()
    tk = tqdm(dataloaderDev, total=len(dataloaderDev), position=0, leave=True)

    with torch.no_grad():
        for data in tqdm(tk):
            inputs, decoderInput, labels = data
            predictList, prbList = model.generate(inputs, maxLen=64)

            predict = decode(predictList, id2word)
            target = decode(labels.cpu().squeeze().tolist(), id2word)
            print("----------")
            print(predict)
            print(target)
            if len(predict) == 0:
                bleu.update(0.0, inputs.shape[0])
            else:
                bleu.update(bleu_score(predict, target), inputs.shape[0])
            tk.set_postfix(bleu=bleu.avg)
    return bleu.avg

def bleu_score(predict, target):
    predict = [item for item in predict]
    target = [item for item in target]
    return bleu.sentence_bleu(predict, target, weights=[1])


if __name__ == "__main__":
    #古诗
    model.load_state_dict(torch.load("./checkpoint/poemSeq2Seq.pt",map_location=device))
    # 对联
    # model.load_state_dict(torch.load("./checkpoint/seq2seq0.25816018783153544.pt",map_location=device))
    generate("瑟批")
    # train()
