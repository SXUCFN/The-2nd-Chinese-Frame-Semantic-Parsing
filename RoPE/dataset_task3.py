import torch
import codecs
import json
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i
        self.data = []
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            cfn_spans = line["cfn_spans"]
            for spans in cfn_spans:
                if spans["end"] + 1 < target[0]:
                    label_idx = [spans["start"] + 1, spans["end"] + 1]
                elif spans["start"] + 1 > target[1]:
                    label_idx = [spans["start"] + 3, spans["end"] + 3]
                fe_text = text[spans["start"]: spans["end"] + 1]
                self.data.append({
                    'text': text,
                    "label_class": self.label2idx[spans["fe_name"]],
                    "label_idx": label_idx,
                    "sentence_id": line["sentence_id"],
                    "target": target,
                    "fe_text": fe_text
                })
        self.num_labels = len(self.idx2label)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        label = d1["label_class"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, label_idx, label, sentence_id


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]




