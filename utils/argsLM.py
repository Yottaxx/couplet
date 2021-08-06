import argparse


def getArgsLM():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./data/condition_target_ner/raw_data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default='./output/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=16,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--embedding_dim",
                        default=256,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--hidden_dim",
                        default=512,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_layers",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--dropout",
                        default=0.3,
                        type=float,
                        help="Total number of training epochs to perform.")
    args = parser.parse_args()
    return args

