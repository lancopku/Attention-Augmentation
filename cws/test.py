import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from pytorch_pretrained_bert import BertAdam
from network import Network
import transformers
from tokenizations import tokenization_bert
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from prepare import build_files
from viterbi import evaluate
from IPython import embed

def random_word(tokens, tokenizer):
    output = []
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob<0.8:
                tokens[i] = tokenizer.mask_token_id
            elif prob<0.9:
                tokens[i] = random.randrange(len(tokenizer.vocab))
            output.append(token)
        else:
            output.append(-100)
    return tokens, output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='raw_data/news_train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--en_path', default='raw_data/news_en.pkl', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=20, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=20, type=int, required=False, help='训练batch size')
    parser.add_argument('--unsupervised_batch_size', default=20, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=2e-5, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=2000, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=1, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())


    #model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    #print('config:\n' + model_config.to_json_string())

    #full_tokenizer = tokenization_bert.BertTokenizer('./cache/vocab.txt')
    full_tokenizer = tokenization_bert.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    full_tokenizer.max_len = 999999
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    en_path = args.en_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    unsupervised_batch_size = args.unsupervised_batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Network.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    #model = Network.from_pretrained('bert-base-chinese', num_labels=4)
    model.load_state_dict(torch.load('model/final_model/model.pt'))
    model.to(device)

    multi_gpu = False
    full_len = 0
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
        
    model.eval()
    #result_news = evaluate('raw_data/news_test.txt', 'raw_data/news_en.pkl', 'raw_data/news_words.txt', model, full_tokenizer, batch_size)
    #print('news results:')
    #print(result_news)
    #result_patent = evaluate('raw_data/patent_test.txt', 'raw_data/patent_en.pkl', 'raw_data/patent_words.txt', model, full_tokenizer, batch_size, output_name='output/patent_fake.txt')
    result_patent = evaluate('raw_data/patent_train.txt', 'raw_data/patent_en.pkl', 'raw_data/patent_words.txt', model, full_tokenizer, batch_size, output_name='output/patent_fake.txt')
    print('patent results:')
    print(result_patent)

if __name__ == '__main__':
    main()
