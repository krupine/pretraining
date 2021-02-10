# -*- coding: utf-8 -*-
'''
Created on 2020. 11. 6.

@author: krupine
'''
import json
import glob
import os
import re
import pickle
import ray
from kss import split_sentences
from soynlp.normalizer import repeat_normalize


@ray.remote
def preprocessing(lines):
        
    p = re.compile(r'[^ .,?!/@$%~％·∼()[]a-zA-Z가-힣一-龥0-9]+')
    text_list = []

    for line in lines:
        sub_text = p.sub('', line)
        sub_text = repeat_normalize(sub_text, num_repeats=2)
        split_sentences_list = split_sentences(sub_text, safe=True)
        
        for sentence in split_sentences_list:
            if len(sentence) < 10:
                continue
            else:
                text_list.append(sentence)
    
    return text_list

def main(num_cpus):

    root_dir = './data/pretrain_data/'

    files = os.listdir(root_dir)
    
    count = 0
    doc_num = 0
    
    merged_file_name = './data/pretrain_data_v2/preprocessed_text.txt'

    text_list = []

    f = open(root_dir + files[0], 'r', encoding='utf-8')

        
    lines = f.readlines()
    line_id = ray.put(lines)

    text_list = ray.get([preprocessing.remote(line_id) for i in range(num_cpus)])
        
    f.close()

    with open(merged_file_name, 'wb', encoding='utf-8') as foo:
        pickle.dump(text_list, foo)

        
if __name__ == '__main__':
    num_cpus = 4
    ray.init(num_cpus = num_cpus, ignore_reinit_error=True)

    main(num_cpus)
#     json_parsing('./data/pretrain_data/news_json/')