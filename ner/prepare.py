import json
import os
import torch
import random
from tqdm import tqdm
from IPython import embed

id2tag = ['O']
for prefix in ['B-', 'I-', 'E-', 'S-']:
    for suffix in ['nr', 'ns', 'nt']:
        id2tag.append(prefix+suffix)
tag2id = {j:i for i,j in enumerate(id2tag)}

def preprocess(s, full_tokenizer):
    lst = s.strip().split()
    words = []
    tags = []
    for tmp in lst:
        word, tag = tmp.split('/')
        for w in word:
            words.append(w)
        if tag=='O':
            for w in word:
                tags.append('O')
        else:
            if len(word)==1:
                tags.append('S-'+tag)
            else:
                tags.append('B-'+tag)
                for i in range(len(word)-2):
                    tags.append('I-'+tag)
                tags.append('E-'+tag)
    data = []
    for word, tag in zip(words, tags):
        data.append((word, tag2id[tag]))
    return list(zip(*data))

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

def make_batches(datas, unsupervised_datas, batch_size, full_tokenizer, train=True):
    if train:
        random.shuffle(datas)
        random.shuffle(unsupervised_datas)
    n_steps = len(datas)//batch_size
    if not train and len(datas)%batch_size!=0:
        n_steps += 1
    if train:
        u_now = 0
    step = 0
    while True:
        if train:
            if step>=n_steps:
                break
            batch = datas[step*batch_size:min((step+1)*batch_size, len(datas))]
            step += 1
            if u_now+batch_size>len(unsupervised_datas):
                random.shuffle(unsupervised_datas)
                u_now = 0
            batch_unsupervised = unsupervised_datas[u_now:u_now+batch_size]
            u_now += batch_size
            batch = batch + batch_unsupervised
        else:
            if step>=len(datas):
                break
            batch = []
            start_step = step
            while step<len(datas) and step-start_step<batch_size:
                start = 0
                while start<len(datas[step][0]):
                    end = min(start+128, len(datas[step][0]))
                    batch.append((datas[step][0][start:end], datas[step][1][start:end], datas[step][2][start:end]))
                    start+=128
                step += 1
        batch_inputs = []
        batch_segments = []
        batch_tags = []
        attention_masks = []
        if train:
            batch_lm_lbls = []
        maxlen = max([len(x)+len(y)+1 for x,y,z in batch])
        maxx = 128
        if train:
            maxlen = max([min(len(x),maxx)+len(y)+1 for x,y,z in batch])
        for ids, ens, tags in batch:
            int_ids = [int(x) for x in ids]
            int_ens = [int(x) for x in ens]
            if tags is not None:
                int_tags = [int(x) for x in tags]
            else:
                int_tags = [-100 for _ in range(len(ids))]
            if len(int_ids)>maxx:
                int_ids = int_ids[:maxx]
                int_tags = int_tags[:maxx]
            if train:
                int_ids, raw_lbls = random_word(int_ids, full_tokenizer)
                int_ens, en_lbls = random_word(int_ens, full_tokenizer)
            input_ids = int_ids+[full_tokenizer.sep_token_id]+int_ens
            segment_ids = [0 for _ in range(len(int_ids)+1)] + [1 for _ in range(len(int_ens))]
            padnum = maxlen-len(input_ids)
            input_ids.extend([full_tokenizer.pad_token_id for _ in range(padnum)])
            segment_ids.extend([full_tokenizer.pad_token_id for _ in range(padnum)])
            real_tags = int_tags+[-100 for _ in range(len(int_ens)+padnum+1)]
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

def en_prepare(en_sens):
    for i in range(len(en_sens)):
        s = en_sens[i]
        s = s.strip().replace(' @-@ ','-')
        s = s.replace('&quot;', '"')
        s = s.replace(' <unk> ', ' [UNK] ')
        s = s.replace('\n<unk> ', '\n[UNK] ')
        s = s.replace(' <unk>\n', ' [UNK]\n')
        en_sens[i] = s

def build_files(data_path, tokenized_data_path, en_path, 
        unsupervised_path, unsupervised_en_path, num_pieces, full_tokenizer):
    with open(data_path, 'r') as f:
        print('reading')
        all_lines = f.readlines()
    with open(en_path) as f:
        en_sens = f.readlines()
    en_prepare(en_sens)
    datas = []
    for s, en in zip(all_lines, en_sens):
        if len(s.strip())==0:
            continue
        data = preprocess(s, full_tokenizer)
        en_data = full_tokenizer.tokenize(en)
        datas.append([*data, en_data])
    all_len = len(datas)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = datas[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(datas[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [(full_tokenizer.convert_tokens_to_ids(line), tag,
            full_tokenizer.convert_tokens_to_ids(en)) for line,tag,en in sublines]
        with open(tokenized_data_path + 'tokenized_train_tok{}.txt'.format(i), 'w') as ftok:
            with open(tokenized_data_path + 'tokenized_train_entok{}.txt'.format(i), 'w') as fentok:
                with open(tokenized_data_path + 'tokenized_train_tag{}.txt'.format(i), 'w') as ftag:
                    for tokens, tag, entok in sublines:
                        ftok.write(' '.join(map(str, tokens))+'\n')
                        fentok.write(' '.join(map(str, entok))+'\n')
                        ftag.write(' '.join(map(str, tag))+'\n')
    
    print('preparing unsupervised data')
    datas = []
    for u_data_path, u_en_path in zip(unsupervised_path, unsupervised_en_path):
        with open(u_data_path) as f:
            all_lines = f.readlines()
        with open(u_en_path) as f:
            en_sens = f.readlines()
        en_prepare(en_sens)
        for s, en in zip(all_lines, en_sens):
            data = preprocess(s, full_tokenizer)
            en_data = full_tokenizer.tokenize(en)
            datas.append([data[0], en_data])
    for i in tqdm(range(num_pieces)):
        sublines = datas[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(datas[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [(full_tokenizer.convert_tokens_to_ids(line),
            full_tokenizer.convert_tokens_to_ids(en)) for line,en in sublines]
        with open(tokenized_data_path + 'tokenized_unsupervised_tok{}.txt'.format(i), 'w') as ftok:
            with open(tokenized_data_path + 'tokenized_unsupervised_entok{}.txt'.format(i), 'w') as fentok:
                for tokens, entok in sublines:
                    ftok.write(' '.join(map(str, tokens))+'\n')
                    fentok.write(' '.join(map(str, entok))+'\n')

    print('finish')

