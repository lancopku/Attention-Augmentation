from prepare import preprocess, make_batches, id2tag, tag2id, en_prepare
import tempfile
import torch
from IPython import embed
import numpy as np
from tqdm import tqdm
import subprocess

tag_transfer = np.zeros((13,13))
start_tags = set()
end_tags = set()
for i in range(13):
    stag = id2tag[i]
    if stag=='O' or stag.startswith('E-') or stag.startswith('S-'):
        end_tags.add(i)
    if stag=='O' or stag.startswith('B-') or stag.startswith('S-'):
        start_tags.add(i)
    for j in range(13):
        ttag = id2tag[j]
        if stag=='O' or stag.startswith('E-') or stag.startswith('S-'):
            if ttag=='O' or ttag.startswith('S-') or ttag.startswith('B-'):
                tag_transfer[i,j]=1
        elif stag.startswith('B-') or stag.startswith('I-'):
            if ttag.startswith('I-') or ttag.startswith('E-'):
                if stag[2:]==ttag[2:]:
                    tag_transfer[i,j]=1
        else:
            raise Exception('Unknown tag')

def decode_viterbi(logprob, ignore):
    N = logprob.shape[1]
    result = []
    for logp, ign in zip(logprob, ignore):
        lbls = np.zeros(N, dtype=np.int32)
        last = np.zeros((N,13), dtype=np.int32)
        stat = np.zeros((N,13))-10000
        for t in start_tags:
            stat[0,t] = logp[0,t]
        for i in range(1, N):
            for j in range(13):
                if ign[i]:
                    stat[i,j]=stat[i-1][j]
                    last[i,j]=j
                    continue
                for k in range(13):
                    if not tag_transfer[j,k]:
                        continue
                    val = stat[i-1,j]+logp[i,k]
                    if val>stat[i,k]:
                        stat[i,k] = val
                        last[i,k] = j
        now = N-1
        ma = -10000
        for t in end_tags:
            if stat[now,t]>ma:
                ma = stat[now,t]
                l = t
        lbls[now] = l
        while now>0:
            l = last[now, l]
            now -= 1
            lbls[now] = l
        tags = []
        for i in range(N):
            if not ign[i]:
                tags.append(id2tag[lbls[i]])
        result.append(tags)
    return result

def sen2tag(s):
    ret = []
    for chunk in s:
        word, tag = chunk.split('/')
        for idx, cha in enumerate(word):
            if tag=='O':
                gold = 'O'
            elif idx==0:
                gold = 'B-'+tag
            else:
                gold = 'I-'+tag
            ret.append((cha, gold))
    return ret

def evaluate(gold_file, en_file, model, tokenizer, batch_size, output_name=None):
    print('testing %s'%gold_file)
    with open(gold_file) as f:
        all_lines = f.readlines()
    if output_name is None:
        fout = tempfile.NamedTemporaryFile('w')
        output_name = fout.name
    else:
        fout = open(output_name, 'w')
    with open(en_file) as f:
        en_sens = f.readlines()
    en_prepare(en_sens)
    datas = []
    tokens = []
    for s, en in zip(all_lines, en_sens):
        if len(s.strip())==0:
            continue
        data = preprocess(s, tokenizer)
        datas.append(data)
        en_tok = tokenizer.tokenize(en)
        tokens.append((tokenizer.convert_tokens_to_ids(data[0]), 
            tokenizer.convert_tokens_to_ids(en_tok), data[1]))

    now = 0
    pos = 0
    s = sen2tag(all_lines[now].strip().split())
    #bar = tqdm(total=len(tokens),ascii=True)
    for x, segment_ids, tag, attention_mask in make_batches(tokens, None, batch_size, tokenizer, train=False):
        output_logits = model(x, token_type_ids=segment_ids,
                attention_mask=attention_mask)
        ignore = (tag==-100) | (x==0)
        lbls = decode_viterbi(output_logits.cpu().data.numpy(), ignore.cpu().data.numpy())
        for i in range(len(lbls)):
            l = lbls[i]
            assert pos+len(l)<=len(s)
            for idx in range(len(l)):
                cha, gold = s[pos+idx]
                pred = l[idx]
                if pred.startswith('E-'):
                    predout = 'I-'+pred[2:]
                elif pred.startswith('S-'):
                    predout = 'B-'+pred[2:]
                else:
                    predout = pred
                fout.write('%s %s %s\n'%(cha,gold,predout))
            if pos+len(l)==len(s):
                now += 1
                #bar.update(1)
                pos = 0
                if now<len(all_lines):
                    s = sen2tag(all_lines[now].strip().split())
            else:
                pos = pos+len(l)
            fout.write('\n')
        # bar.update(len(lbls))
    fout.flush()
    
    command = './conlleval'
    with open(output_name) as fin:
        out = subprocess.Popen(command.split(' '), stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    line = stdout.split('\n')[1]
    P = float(line.split('precision:')[-1].split('%')[0].strip())
    R = float(line.split('recall:')[-1].split('%')[0].strip())
    F = float(line.split('FB1:')[-1].split('%')[0].strip())
    result = {'Precision':P, 'Recall':R, 'F1':F}

    fout.close()
    return result
