import json
from openai import OpenAI
from utils import *
import os
import sys
from tqdm import tqdm

# 根据网络环境设置代理
os.environ['HTTP_PROXY'] = 'http://xxxx:xxxx'
os.environ['HTTPS_PROXY'] = 'http://xxxx:xxxx'

# 根据实际情况填写api_key
client = OpenAI(api_key='sk-xxxxxx')

datas = json.load(open('./data/cfn-test-A.json'))
frame_set, fe_set = get_frame_info()

task = sys.argv[1]
assert task in ['AI', 'RI', 'FI'], "Task must be one of ['AI', 'RI', 'FI']!"

results = []
for data in tqdm(datas):
    sentence = data['text']
    sentence_id = data['sentence_id']
    for target_index in data['target']:
        target_start, target_end = target_index['start'], target_index['end']+1
        target = sentence[target_start:target_end]
        if task == 'AI':
            result = AI(client, sentence, target, sentence_id)
        elif task == 'RI':
            result = RI(client, sentence, target, fe_set, sentence_id)
        elif task == 'FI':
            result = FI(client, sentence, sentence_id)
        results.extend(result)
        json.dump(results, open(f'./data/{task}_results.json', 'w'))