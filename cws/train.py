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
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from prepare import build_files, make_batches
from viterbi import evaluate
from IPython import embed


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
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        unsupervised = ['raw_data/medical_train.txt', 'raw_data/news_test.txt']
        unsupervised_en = ['raw_data/medical_en.pkl', 'raw_data/news_en.pkl']
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, en_path=en_path, 
                    num_pieces=num_pieces, unsupervised=unsupervised, unsupervised_en=unsupervised_en,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    model = Network.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    #model = Network.from_pretrained('bert-base-chinese', num_labels=4)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_tag{}.txt'.format(i), 'r') as f:
            lines = f.readlines()
            full_len += len(lines)
    total_steps = int(full_len * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = BertAdam(model.parameters(), lr=lr)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
        
    print('starting training')
    overall_step = 0
    for epoch in range(epochs):
        running_loss = 0
        running_loss_tag = 0
        running_loss_lm = 0
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        model.train()
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_tok{}.txt'.format(i), 'r') as f:
                tokens = f.readlines()
            with open(tokenized_data_path + 'tokenized_train_entok{}.txt'.format(i), 'r') as f:
                ens = f.readlines()
            with open(tokenized_data_path + 'tokenized_train_tag{}.txt'.format(i), 'r') as f:
                tags = f.readlines()
            with open(tokenized_data_path + 'tokenized_unsupervised_tok{}.txt'.format(i), 'r') as f:
                u_tokens = f.readlines()
            with open(tokenized_data_path + 'tokenized_unsupervised_entok{}.txt'.format(i), 'r') as f:
                u_ens = f.readlines()
            
            samples = [(x.strip().split(),y.strip().split(),z.strip().split()) for x,y,z in zip(tokens, ens, tags)]
            u_samples = [(x.strip().split(),y.strip().split(),None) for x,y in zip(u_tokens, u_ens)]
            random.shuffle(samples)
            random.shuffle(u_samples)
            u_now = 0
            for step, (batch_inputs, batch_segments, batch_tags, attention_mask, batch_lm_lbls) in enumerate(make_batches(samples, u_samples, batch_size, full_tokenizer)):  # drop last

                #  forward pass
                loss_tag, loss_lm = model.forward(input_ids=batch_inputs, 
                    token_type_ids=batch_segments, labels=batch_tags,
                    masked_lm_labels=batch_lm_lbls, attention_mask=attention_mask)
                #  get loss
                if multi_gpu:
                    loss_tag = loss_tag.sum()
                    loss_lm = loss_lm.sum()
                loss_tag = loss_tag / (batch_tags!=-100).sum()
                loss_lm = loss_lm / (batch_lm_lbls!=-100).sum()
                loss = loss_tag + loss_lm

                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    running_loss_tag += loss_tag.item()
                    running_loss_lm += loss_lm.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (step + 1) % log_step == 0:
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {:.6f}, tag loss {:.6f}, lm loss {:.6f}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation),
                        running_loss_tag * gradient_accumulation / (log_step / gradient_accumulation),
                        running_loss_lm * gradient_accumulation / (log_step / gradient_accumulation)
                        ))
                    running_loss = 0
                    running_loss_tag = 0
                    running_loss_lm = 0
                overall_step += 1
            piece_num += 1

        model.eval()
        result_news = evaluate('raw_data/news_test.txt', 'raw_data/news_en.pkl', 'raw_data/news_words.txt', model, full_tokenizer, batch_size)
        print('news results:')
        print(result_news)
        result_medical = evaluate('raw_data/medical_test.txt', 'raw_data/medical_en.pkl', 'raw_data/medical_words.txt', model, full_tokenizer, batch_size)
        print('medical results:')
        print(result_medical)
        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        #model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        torch.save(model_to_save.state_dict(), output_dir + 'model_epoch{}/model.pt'.format(epoch+1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.save_pretrained(output_dir + 'final_model')
    torch.save(model_to_save.state_dict(), output_dir + 'final_model/model.pt')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
