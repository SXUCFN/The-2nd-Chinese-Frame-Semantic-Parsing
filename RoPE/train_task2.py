#!/usr/bin/python3

import os
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from dataset_task2 import Dataset
from params import args
from model_task2 import Model


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def get_model_input(data, device=None):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    target_cls = []
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])
        target_cls.append(d[5])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)
    target_cls = np.array(target_cls, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)
    target_cls = torch.from_numpy(target_cls).to(device)

    H_label = torch.zeros(bs, max_len, max_len).to(device)
    for i in range(bs):
        for idx in labels[i]:
            H_label[i][idx[0], idx[1]] = 1

    return input_ids, attention_mask, target, H_label, sentence_id, target_cls


def eval(model, val_loader):
    model.eval()
    H_correct = 0.0
    H_precision_total = 0.0
    H_recall_total = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, label, sentence_id, target_cls = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                           device=device, for_test=True)

            # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)

            H_attention_mask = torch.triu(
                torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
            H_pred = torch.where(
                output["logits"] >= 0,
                torch.ones(output["logits"].shape).to(device),
                torch.zeros(output["logits"].shape).to(device)
            ) * H_attention_mask


            # H_pred = torch.argmax(F.softmax(output["logits"], dim=-1), dim=-1)

            H_correct += ((H_pred == label) * (label != 0)).sum().item()
            H_precision_total += (H_pred != 0).sum().item()
            H_recall_total += (label != 0).sum().item()

    H_precision = H_correct / (H_precision_total + 1e-6)
    H_recall = H_correct / (H_recall_total + 1e-6)


    return H_precision, H_recall


def train(model, train_loader, val_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # *******************  NoisyTune  ****************
    noise_lambda = 0.15
    for name, para in param_optimizer:
        model.state_dict()[name][:] += \
            (torch.rand(para.size()).to(device) - 0.5) * \
            noise_lambda * torch.std(para)

    # ***********************************************

    total_steps = int(len(train_loader) * args.num_train_epochs /
                      args.accumulate_gradients)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr)

    # adv = FGM(bert_model) if args.with_adv_train else None
    global_step = 0
    best_f1 = 0.0
    fgm = FGM(model)
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            global_step += 1

            input_ids, attention_mask, target, label, sentence_id, target_cls = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                           device=device)

            # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)

            loss = output['loss']

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')

            loss.backward()

            fgm.attack()  # embedding被修改了
            # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
            loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=label,
                               device=device)[
                    'loss']
            # loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)["loss"]
            loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复Embedding的参数

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * \
                               warmup_linear(global_step / total_steps,
                                             args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # break

        H_precision, H_recall = eval(model, val_loader)
        f1 = 2 * H_precision * H_recall / (H_precision + H_recall + 1e-6)
        if f1 > best_f1:
            print(f'saved! new best f1 {f1}, ori_f1 {best_f1}')
            print(f"H_precision: {H_precision}, H_recall: {H_recall}")
            best_f1 = f1
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs('saves', exist_ok=True)
            torch.save(model_to_save.state_dict(), f'saves/model_task2_best.bin')
        else:
            print(f'current f1: {f1}')
            print(f" H_precision: {H_precision}, H_recall: {H_recall}")

        # train_loader.dataset.gen_data()


def load_pretrained_bert(bert_model, init_checkpoint):
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', '').replace('roformer.', ''): v for k, v in state.items() if
                     not k.startswith('cls.')}
            # state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=False)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    train_dataset = Dataset("./dataset/cfn-train.json",
                            "./dataset/frame_info.json",
                            tokenizer)
    dev_dataset = Dataset("./dataset/cfn-dev.json",
                          "./dataset/frame_info.json",
                          tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = train_dataset.max_cls
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)
    args.num_train_epochs = 5
    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=True
    )

    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    train(model, train_loader, val_loader)
