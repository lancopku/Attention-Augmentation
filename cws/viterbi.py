from prepare import preprocess, sen2woseg, make_batches
import tempfile
import torch
from IPython import embed
import numpy as np
from tqdm import tqdm
import subprocess
import pickle as pkl

def decode_viterbi(logprob, ignore):
    N = logprob.shape[1]
    viterbi_transform = [[0,1], [2,3], [2,3], [0,1]]
    result = []
    for logp, ign in zip(logprob, ignore):
        lbls = np.zeros(N, dtype=np.int32)
        last = np.zeros((N,4), dtype=np.int32)
        stat = np.zeros((N,4))-10000
        stat[0][0] = logp[0][0]
        stat[0][1] = logp[0][1]
        for i in range(1, N):
            for j in range(4):
                if ign[i]:
                    stat[i,j]=stat[i-1][j]
                    last[i,j]=j
                for k in viterbi_transform[j]:
                    val = stat[i-1,j]+logp[i,k]
                    if val>stat[i,k]:
                        stat[i,k] = val
                        last[i,k] = j
        now = N-1
        if stat[now][3]>stat[now][0]:
            l = 3
        else:
            l = 0
        lbls[now] = l
        while now>0:
            l = last[now, l]
            now -= 1
            lbls[now] = l
        tags = []
        for i in range(N):
            if not ign[i]:
                tags.append(lbls[i])
        result.append(tags)
    return result

def get_words(raw, lbl):
    ret = []
    w = ''
    for i in range(len(lbl)):
        if lbl[i]==0:
            assert w==''
            ret.append(raw[i])
        elif lbl[i]==1:
            assert w==''
            w=raw[i]
        else:
            assert w!=''
            w=w+raw[i]
            if lbl[i]==3:
                ret.append(w)
                w = ''
    assert w==''
    return ret

def evaluate(gold_file, en_file, word_file, model, tokenizer, batch_size, output_name=None):
    print('testing %s'%gold_file)
    with open(gold_file, encoding='utf-8') as f:
        lines = f.readlines()
    with open(en_file, 'rb') as f:
        mp = pkl.load(f)
    if output_name is None:
        fout = tempfile.NamedTemporaryFile('w', encoding='utf-8')
        output_name = fout.name
    else:
        fout = open(output_name, 'w', encoding='utf-8')
    datas = []
    all_lbls = []
    tokens = []
    for line in lines:
        woseg = sen2woseg(line.strip())
        parts, raw_parts = preprocess(woseg, addpunc=True)
        tmp = []
        for part, raw in zip(parts, raw_parts):
            tmp.append((part, raw))
            if len(part)<=1:
                continue
            en = mp[''.join(raw)][0]
            en_tok = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(en))
            token = tokenizer.convert_tokens_to_ids(part)
            tokens.append((token, en_tok, [0 for _ in range(len(token))]))
        datas.append(tmp)

    #bar = tqdm(total=len(tokens))
    for x, segment_ids, tag, attention_mask in make_batches(tokens, None, batch_size, tokenizer, train=False):
        output_logits = model(x, token_type_ids=segment_ids, attention_mask=attention_mask).cpu().data.numpy()
        ignore = ((tag==-100) | (x==0)).cpu().data.numpy()
        lbls = decode_viterbi(output_logits, ignore)
        all_lbls.extend(lbls)
        #bar.update(len(lbls))
      
    now = 0
    for sen in datas:
        out = []
        for part, raw in sen:
            if len(part)<=1:
                out.extend(raw)
                continue
            out.extend(get_words(raw, all_lbls[now]))
            now += 1
        fout.write(' '.join(out)+'\n')
    fout.flush()
    
    command = 'perl score.pl %s %s %s'%(word_file, gold_file, output_name)
    out = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    for line in stdout.split('\n'):
        if line.startswith('=== F MEASURE'):
            F_score = float(line.split('\t')[-1])

    fout.close()
    return F_score
