import json

def FI(client, sentence, target, frame_set, sentence_id):
    prompt = f"给定一个句子：\n{sentence}\n其中的目标词为“{target}”。根据给定的句子和其中的目标词在句子中语义场景，在我给定的候选框架中为目标词选择一个最合适的框架。候选框架包括：{frame_set}\n只需要回答框架名称，不要添加其他内容。"
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system",
                "content": "你是一个框架语义学家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    res = completion.choices[0].message.content
    return [[sentence_id, res]]

def AI(client, sentence, target, sentence_id):
    prompt = f"给定一个句子：\n{sentence}\n其中的目标词为“{target}”。根据我给定的句子和其中的目标词在句子中语义场景，找出属于目标词的所有论元。只需要以Python数组的形式输出所有论元，不要添加其他内容。"
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system",
                "content": "你是一个框架语义学家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    res = completion.choices[0].message.content
    result = []
    try:
        items = json.loads(res[res.find('['):res.rfind(']')+1])
        for item in items:
            result.append([sentence_id, sentence.index(item), sentence.index(item)+len(item)-1])
    except:
        pass
    return result

def RI(client, sentence, target, fe_set, sentence_id):
    prompt = f"给定一个句子：\n{sentence}\n其中的目标词为“{target}”。根据我给定的句子和其中的目标词在句子中语义场景，在句子中找到所有符合候选框架元素的论元，并完成论元与框架元素的匹配。候选框架元素集合如下：{fe_set}。\n只需要以Json键值对的形式输出结果，其中键为句子中的论元原文，值为论元对应的框架元素名称，不要添加其他内容。"
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system",
                "content": "你是一个框架语义学家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    res = completion.choices[0].message.content
    result = []
    try:
        items = json.loads(res[res.find('{'):res.rfind('}')+1])
        for fe_name in items:
            result.append([sentence_id, sentence.index(items[fe_name]), sentence.index(items[fe_name])+len(items[fe_name])-1, fe_name])
    except:
        pass
    return result

def get_frame_info():
    frame_info = json.load(open('./data/frame_info.json'))
    fe_set = set()
    frame_set = set()
    for info in frame_info:
        fe_set.add(info['frame_name'])
        for fe in info['fes']:
            frame_set.add(fe['fe_name'])
    return frame_set, fe_set