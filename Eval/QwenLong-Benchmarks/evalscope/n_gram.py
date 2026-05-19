# N-gram 数据污染检测脚本
#
# 用法:
#   python n_gram.py \
#       --tokenizer_bin  /path/to/cstokenizer \
#       --tokenizer_model /path/to/tokenizer.model \
#       --to_validate_file /path/to/train.jsonl \
#       --outfile_prefix   /path/to/output/prefix \
#       --test_benchmark_lst /path/to/eval.lst \
#       [--ngrams 9 13 30] [--stop_words_file /path/to/patterns.txt] [--num_workers 8]
#
# eval.lst 每行一个测试集 JSONL 文件路径。
# 输出：<prefix>_rm_<n>（干净数据）、<prefix>.match_<n>（命中详情）、<prefix>.hit_<n>（命中原始样本）

import sys
import json
import re
import os
import unicodedata
import pickle
import argparse
import multiprocessing
import subprocess

from tqdm import tqdm



class Ngram_decontamination():

    def __init__(self, tokenizer_bin, tokenizer_model, ngram=13, stop_words_file=None):
        self.tokenizer = subprocess.Popen(
            [tokenizer_bin, '--encode', tokenizer_model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.id2token, self.token2id = self.read_id2token_map(tokenizer_model)
        self.ngram = ngram
        self.ngram2q = {}
        self.ngram2source = {}
        self.ngram2text_id = {}
        self.id2text = {}
        self.id2source = {}
        self._text_id_counter = 0

        self.stop_words_ = []
        if stop_words_file and os.path.exists(stop_words_file):
            with open(stop_words_file, 'r') as f:
                for line in f:
                    self.stop_words_.append(line.rstrip('\n'))
        self.stop_words = [x.encode().decode('unicode_escape') for x in self.stop_words_]

    def read_id2token_map(self, tokenizer_model):
        id2token = {}
        token2id = {}
        with open(tokenizer_model, encoding='utf-8-sig') as f:
            json_dict = json.load(f)
            for one in json_dict["CommonTokens"]:
                id2token[one['TokenID']] = bytes(one['TokenBytes'])
                token2id[bytes(one['TokenBytes'])] = one['TokenID']
            for one in json_dict["SpecialTokens"]:
                id2token[one['TokenID']] = one['TokenStr'].encode('utf8')
                token2id[one['TokenStr'].encode('utf8')] = one['TokenID']
        return id2token, token2id

    def text2token(self, text):
        if len(text) == 0:
            return []
        tk_j = {'text': text}
        s = str(json.dumps(tk_j, ensure_ascii=False))
        self.tokenizer.stdin.write(s + "\n")
        self.tokenizer.stdin.flush()
        segs = self.tokenizer.stdout.readline().strip().split(',')
        segs = [int(one) for one in segs]
        pieces = [self.id2token[one] for one in segs]
        return pieces

    def text2substring(self, text):
        if len(text) == 0:
            return []
        tokens = self.text2token(text)
        return [x.decode('utf-8', errors='replace') for x in tokens]

    def load_cache(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 5:
                    self.ngram2source, self.ngram2q, self.ngram2text_id, self.id2text, self.id2source = data
                elif len(data) == 4:
                    self.ngram2source, self.ngram2q, self.ngram2text_id, self.id2text = data
                else:
                    self.ngram2source, self.ngram2q = data

    def save_cache(self, cache_file):
        with open(cache_file, 'wb') as f:
            pickle.dump((self.ngram2source, self.ngram2q, self.ngram2text_id, self.id2text, self.id2source), f)

    def store_ngram_dict(self, testfiles):
        cache_file = testfiles + '_' + str(self.ngram) + '.cache'
        if os.path.exists(cache_file):
            self.load_cache(cache_file)
            return

        with open(testfiles, 'r') as reader:
            files = [line.strip() for line in reader if line.strip()]

        for f_name in tqdm(files):
            with open(f_name) as f:
                for line in f:
                    line = line.rstrip()
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    text = j.get("text", "")
                    if not text:
                        continue
                    split_content = self.text2substring(text[:4000])
                    text_id = self._text_id_counter
                    self.id2text[text_id] = text
                    self.id2source[text_id] = j.get("source", "")
                    self._text_id_counter += 1
                    for i in range(len(split_content) - self.ngram + 1):
                        span = ''.join(split_content[i:i+self.ngram])
                        self.ngram2source[span] = f_name
                        self.ngram2text_id[span] = text_id
        self.save_cache(cache_file)

    def check_stop_words(self, text):
        for word in self.stop_words:
            if text.lower() in word.lower() or word.lower() in text.lower():
                return True
        return text in self.stop_words

    def check_single_line(self, text):
        split_content = self.text2substring(text)

        if len(split_content) >= self.ngram:
            for i in range(len(split_content) - self.ngram + 1):
                span = ''.join(split_content[i:i+self.ngram])
                if span in self.ngram2source and not self.check_stop_words(span):
                    text_id = self.ngram2text_id.get(span, -1)
                    return {
                        'hit': True, 'span': span, 'full_text': text,
                        'span_in_source': self.ngram2source[span],
                        'test_text': self.id2text.get(text_id, ''),
                        'test_text_id': text_id,
                        'test_source': self.id2source.get(text_id, ''),
                    }
            return {'hit': False}
        else:
            span = ''.join(split_content)
            for key in self.ngram2source:
                if span in key:
                    text_id = self.ngram2text_id.get(key, -1)
                    return {
                        'hit': True, 'span': span, 'full_text': text,
                        'span_in_source': self.ngram2source[key],
                        'test_text': self.id2text.get(text_id, ''),
                        'test_text_id': text_id,
                        'test_source': self.id2source.get(text_id, ''),
                    }
            return {'hit': False}



def iter_jsonl_file(infile):
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(infile):
    with open(infile, 'r') as f:
        return sum(1 for line in f if line.strip())



def extract_prompt(item):
    return item['extra_info']['question']


_worker_ngrams = {}



def init_worker(cache_files, ngrams, tokenizer_bin, tokenizer_model, stop_words_file):
    global _worker_ngrams, _test_tokens
    first_dec = Ngram_decontamination(tokenizer_bin, tokenizer_model, ngram=ngrams[0], stop_words_file=stop_words_file)
    first_dec.load_cache(cache_files[0])
    _worker_ngrams[ngrams[0]] = first_dec

    for n, cache_file in zip(ngrams[1:], cache_files[1:]):
        dec = object.__new__(Ngram_decontamination)
        dec.ngram = n
        dec.tokenizer    = first_dec.tokenizer
        dec.id2token     = first_dec.id2token
        dec.token2id     = first_dec.token2id
        dec.stop_words   = first_dec.stop_words
        dec.stop_words_  = first_dec.stop_words_
        dec.ngram2q       = {}
        dec.ngram2source  = {}
        dec.ngram2text_id = {}
        dec.id2text       = {}
        dec.id2source     = {}
        dec._text_id_counter = 0
        dec.load_cache(cache_file)
        _worker_ngrams[n] = dec



def process_item(item):
    qa_text = extract_prompt(item)
    ngram_results = {}
    for n, dec in _worker_ngrams.items():
        hit_result = dec.check_single_line(qa_text)
        hit_result.pop('test_text', None)
        ngram_results[n] = hit_result
    return item, ngram_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngrams', type=int, nargs='+', default=[13],
                        help='n-gram sizes, e.g. --ngrams 9 13 30')
    parser.add_argument('--tokenizer_bin',       type=str, required=True,  help='path to tokenizer binary')
    parser.add_argument('--tokenizer_model',     type=str, required=True,  help='path to tokenizer model file')
    parser.add_argument('--stop_words_file',     type=str, default=None,   help='optional stop-words file (one pattern per line)')
    parser.add_argument('--to_validate_file',    type=str, required=True,  help='training JSONL to check for contamination')
    parser.add_argument('--outfile_prefix',      type=str, required=True,  help='output file prefix')
    parser.add_argument('--test_benchmark_lst',  type=str, required=True,  help='file listing test benchmark JSONL paths')
    parser.add_argument('--num_workers',         type=int, default=8,      help='number of parallel workers')
    args = parser.parse_args()

    ngrams = args.ngrams
    test_benchmark_lst = args.test_benchmark_lst

    print(f'Building ngram dicts for n={ngrams} ...')
    for n in ngrams:
        dec = Ngram_decontamination(args.tokenizer_bin, args.tokenizer_model,
                                    ngram=n, stop_words_file=args.stop_words_file)
        dec.store_ngram_dict(test_benchmark_lst)

    to_validate_file = args.to_validate_file
    outfile_prefix   = args.outfile_prefix
    tot = count_lines(to_validate_file)

    cache_files = [test_benchmark_lst + '_' + str(n) + '.cache' for n in ngrams]
    os.makedirs(os.path.dirname(outfile_prefix), exist_ok=True)

    stats_by_n = {n: {'match': 0, 'hit_test_ids': set()} for n in ngrams}

    out_files = {}
    for n in ngrams:
        out_files[n] = {
            'cleaned':  open(outfile_prefix + f'_rm_{n}',    'w', encoding='utf-8'),
            'hit_data': open(outfile_prefix + f'.match_{n}', 'w', encoding='utf-8'),
            'hit_ori':  open(outfile_prefix + f'.hit_{n}',   'w', encoding='utf-8'),
        }

    try:
        with multiprocessing.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(cache_files, ngrams, args.tokenizer_bin, args.tokenizer_model, args.stop_words_file)
        ) as pool:
            for item, ngram_results in tqdm(
                pool.imap(process_item, iter_jsonl_file(to_validate_file), chunksize=8), total=tot
            ):
                for n, hit_result in ngram_results.items():
                    s = stats_by_n[n]
                    fh = out_files[n]
                    if hit_result['hit']:
                        fh['hit_data'].write(json.dumps(hit_result, ensure_ascii=False) + '\n')
                        fh['hit_ori'].write(json.dumps(item,        ensure_ascii=False) + '\n')
                        s['match'] += 1
                        s['hit_test_ids'].add(hit_result.get('test_text_id', -1))
                    else:
                        fh['cleaned'].write(json.dumps(item, ensure_ascii=False) + '\n')
    finally:
        for n in ngrams:
            for fh in out_files[n].values():
                fh.close()

    tmp_dec = Ngram_decontamination(args.tokenizer_bin, args.tokenizer_model, ngram=ngrams[0])
    tmp_dec.load_cache(cache_files[0])
    total_test_items = len(tmp_dec.id2text)

    for n in ngrams:
        s = stats_by_n[n]
        hit_test  = len(s['hit_test_ids'])
        print(f'ngram={n}:')
        print(f'  contamination rate : {s["match"]/tot:.4f} ({s["match"]}/{tot})')
        print(f'  test set coverage  : {hit_test/total_test_items:.4f} ({hit_test}/{total_test_items})')
