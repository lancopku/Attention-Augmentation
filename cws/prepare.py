import json
import os
import pickle as pkl
from tqdm import tqdm
from IPython import embed
import random
import torch

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

def sen2woseg(line):
    letter = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－'
    line = line.strip()
    line = ' '.join(line.split())
    if len(line)<2:
        return line
    for i in range(1,len(line)-1):
        if line[i]==' ':
            if line[i-1] in letter and line[i+1] in letter:
                line = line[:i]+'@'+line[i+1:]
    line = line.replace(' ','').replace('@',' ')
    return line

def parsetype(T):
    if T=='N':
        return '[NUM]'
    if T=='L':
        return '[LETTER]'
    return '[UNK]'

def preprocess(s, addpunc=False):
    num = '0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％'
    letter = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－'
    punc = ' ，。、：（）；【】～“”《》'

    rawret = []
    rawnow = []
    ret = []
    now = []
    w = ''
    T = 'O'
    for c in s:
        if c in punc:
            if T!='O':
                now.append(parsetype(T))
                rawnow.append(w)
                w = ''
                T = 'O'
            if len(now):
                ret.append(now)
                rawret.append(rawnow)
                now = []
                rawnow = []
            if addpunc and c!=' ':
                ret.append([c])
                rawret.append([c])
            continue
        if c in num:
            if T!='N' and T!='O':
                now.append(parsetype(T))
                rawnow.append(w)
                w = ''
            w = w+c
            T = 'N'
        elif c in letter:
            if T!='L' and T!='O':
                now.append(parsetype(T))
                rawnow.append(w)
                w = ''
            w = w+c
            T = 'L'
        else:
            if T!='O':
                now.append(parsetype(T))
                rawnow.append(w)
                w = ''
                T = 'O'
            now.append(c)
            rawnow.append(c)
    if T!='O':
        now.append(parsetype(T))
        rawnow.append(w)
    if len(now):
        ret.append(now)
        rawret.append(rawnow)
    return ret, rawret

def make_batches(datas, unsupervised_datas, batch_size, full_tokenizer, train=True):
    if train:
        random.shuffle(datas)
        random.shuffle(unsupervised_datas)
    n_steps = len(datas)//batch_size
    if not train and len(datas)%batch_size!=0:
        n_steps += 1
    if train:
        u_now = 0
    for step in range(n_steps):
        batch = datas[step*batch_size:min((step+1)*batch_size, len(datas))]
        if train:
            if u_now+batch_size>len(unsupervised_datas):
                random.shuffle(unsupervised_datas)
                u_now = 0
            batch_unsupervised = unsupervised_datas[u_now:u_now+batch_size]
            u_now += batch_size
            batch = batch + batch_unsupervised
        batch_inputs = []
        batch_segments = []
        batch_tags = []
        attention_masks = []
        if train:
            batch_lm_lbls = []
        maxlen = max([len(x)+len(y)+1 for x,y,z in batch])
        for ids, ens, tags in batch:
            int_ids = [int(x) for x in ids]
            int_ens = [int(x) for x in ens]
            if tags is not None:
                int_tags = [int(x) for x in tags]
            else:
                int_tags = [-100 for _ in range(len(ids))]
            if train:
                int_ids, raw_lbls = random_word(int_ids, full_tokenizer)
                int_ens, en_lbls = random_word(int_ens, full_tokenizer)
            input_ids = int_ids+[full_tokenizer.sep_token_id]+int_ens
            segment_ids = [0 for _ in range(len(int_ids)+1)] + [1 for _ in range(len(int_ens))]
            padnum = maxlen-len(input_ids)
            input_ids.extend([full_tokenizer.pad_token_id for _ in range(padnum)])
            segment_ids.extend([0 for _ in range(padnum)])
            real_tags = int_tags+[-100 for _ in range((len(int_ens)+padnum+1))]
            attention_mask = [1 for _ in range(len(int_ids)+len(int_ens)+1)] + [0 for _ in range(padnum)]
            batch_inputs.append(input_ids)
            batch_segments.append(segment_ids)
            batch_tags.append(real_tags)
            attention_masks.append(attention_mask)
            if train:
                lm_lbls = raw_lbls + [-100] + en_lbls + [-100 for _ in range(padnum)]
                batch_lm_lbls.append(lm_lbls)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        batch_inputs = torch.tensor(batch_inputs).long().to(device)
        batch_segments = torch.tensor(batch_segments).long().to(device)
        batch_tags = torch.tensor(batch_tags).long().to(device)
        attention_masks = torch.tensor(attention_masks).to(device)
        if train:
            batch_lm_lbls = torch.tensor(batch_lm_lbls).long().to(device)
        if train:
            yield batch_inputs, batch_segments, batch_tags, attention_masks, batch_lm_lbls
        else:
            yield batch_inputs, batch_segments, batch_tags, attention_masks



#S:0, B:1, I:2, E:3
def tokenize(lines, mp):
    data = []
    failnum = 0
    for line in lines:
        woseg = sen2woseg(line)
        seg = line.strip().split()
        ret, rawret = preprocess(woseg, True)
        now = 0
        fail = False
        for i in range(len(ret)):
            tmp = rawret[i]
            lbl = []
            start = end = 0
            while end<len(tmp):
                left = len(seg[now])
                while left>0 and end<len(tmp):
                    left -= len(tmp[end])
                    end += 1
                if left!=0:
                    fail = True
                    break
                if end-start==1:
                    lbl.append(0)
                else:
                    lbl.append(1)
                    for k in range(end-start-2):
                        lbl.append(2)
                    lbl.append(3)
                start = end
                now += 1
            if fail:
                failnum += 1
                break
            if len(tmp)>1:
                sen = ''.join(tmp)
                if sen in mp:
                    en = mp[sen][0]
                else:
                    en = None
                data.append((ret[i], en, lbl))
    return data

def build_files(data_path, tokenized_data_path, en_path, 
        unsupervised, unsupervised_en,
        num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = f.readlines()
    with open(en_path, 'rb') as f:
        mp = pkl.load(f)
    datas = tokenize(lines, mp)
    all_len = len(datas)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = datas[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(datas[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        tmp = []
        for line, en, tag in sublines:
            txt_token = full_tokenizer.convert_tokens_to_ids(line)
            tokenized_en = full_tokenizer.tokenize(en)
            en_token = full_tokenizer.convert_tokens_to_ids(tokenized_en)
            tmp.append((txt_token, en_token, tag))
        sublines = tmp
        with open(tokenized_data_path + 'tokenized_train_tok{}.txt'.format(i), 'w') as ftok:
            with open(tokenized_data_path + 'tokenized_train_entok{}.txt'.format(i), 'w') as fen:
                with open(tokenized_data_path + 'tokenized_train_tag{}.txt'.format(i), 'w') as ftag:
                    for tokens, en, tag in sublines:
                        ftok.write(' '.join(map(str, tokens))+'\n')
                        fen.write(' '.join(map(str, en))+'\n')
                        ftag.write(' '.join(map(str, tag))+'\n')
    
    print('preparing unsupervised data')
    datas = []
    for u_data_path, u_en_path in zip(unsupervised, unsupervised_en):
        with open(u_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(u_en_path, 'rb') as f:
            mp = pkl.load(f)
        datas.extend(tokenize(lines, mp))
    for i in tqdm(range(num_pieces)):
        sublines = datas[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(datas[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        tmp = []
        for line, en, _ in sublines:
            txt_token = full_tokenizer.convert_tokens_to_ids(line)
            tokenized_en = full_tokenizer.tokenize(en)
            en_token = full_tokenizer.convert_tokens_to_ids(tokenized_en)
            tmp.append((txt_token, en_token))
        sublines = tmp
        with open(tokenized_data_path + 'tokenized_unsupervised_tok{}.txt'.format(i), 'w') as ftok:
            with open(tokenized_data_path + 'tokenized_unsupervised_entok{}.txt'.format(i), 'w') as fen:
                for tokens, en in sublines:
                    ftok.write(' '.join(map(str, tokens))+'\n')
                    fen.write(' '.join(map(str, en))+'\n')
    print('finish')

