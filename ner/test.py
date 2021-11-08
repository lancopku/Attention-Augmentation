import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,3,5'
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import transformers
from tokenizations import tokenization_bert
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel, CrossEntropyLoss
from prepare import build_files, make_batches
from viterbi import evaluate
from network import Network
from IPython import embed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='raw_data/MSRA_train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--batch_size', default=60, type=int, required=False, help='训练batch size')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--epoch', default=16, type=int)

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
    tokenized_data_path = args.tokenized_data_path
    batch_size = args.batch_size

    model = Network.from_pretrained('bert-base-multilingual-cased', num_labels=17)
    epoch = args.epoch
    model.load_state_dict(torch.load('model/model_epoch%d/model.pt'%epoch))
    model.eval()
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
        
    print('%d:'%epoch)
    #result_test = evaluate('raw_data/MSRA_test.txt', 'raw_data/MSRA_test_en.txt', model, full_tokenizer, batch_size)
    #print('test results:')
    #print(result_test)
    result_cd = evaluate('raw_data/renmin_train.txt', 'raw_data/renmin_train_en.txt', model, full_tokenizer, batch_size, output_name='output/renmin_fake.txt')
    print('cross domain results:')
    print(result_cd)


if __name__ == '__main__':
    main()
