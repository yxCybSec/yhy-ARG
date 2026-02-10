import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

# 标签映射：既支持字符串（"real"/"fake"），也支持数字（0/1），保证数据集里两种写法都能被正确处理
label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

# LLM 预测的标签映射，多了 "other" 这一类
label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2
}


def word2input(texts, max_len, tokenizer):
    """
    将一批文本序列编码成 BERT 可接受的 token_ids 和 attention_mask。
    - texts: list[str]，输入的句子（新闻或 rationale）
    - max_len: 最大序列长度，超出会被截断，不足会 pad
    - tokenizer: 来自 transformers 的 BertTokenizer
    """
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(
                text,
                max_length=max_len,
                add_special_tokens=True,
                padding='max_length',
                truncation=True
            )
        )
    token_ids = torch.tensor(token_ids)

    # attention mask：非 pad 位置为 1，pad 位置为 0
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language):
    """
    读取指定 json 文件，构造一个 PyTorch DataLoader。
    这里只实现了 data_type == 'rationale' 的分支：
    - content: 新闻正文
    - FTR_2 / FTR_3: 两种不同视角的 LLM rationale（例如 top-down / counter-speech）
    - FTR_*_pred: LLM 对 news 的预测（real/fake/other）
    - FTR_*_acc: LLM 预测是否正确（0/1）
    """
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    if data_type == 'rationale':
        # 读取单个月份的 train/val/test.json
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        df_data = pd.DataFrame(columns=('content', 'label'))
        for item in data_list:
            tmp_data = {}

            # === 新闻主体信息 ===
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']

            # === LLM 给出的两类 rationale 以及对应的预测/正确性 ===
            tmp_data['FTR_2'] = item['td_rationale']
            tmp_data['FTR_3'] = item['cs_rationale']

            tmp_data['FTR_2_pred'] = item['td_pred']
            tmp_data['FTR_3_pred'] = item['cs_pred']

            tmp_data['FTR_2_acc'] = item['td_acc']
            tmp_data['FTR_3_acc'] = item['cs_acc']

            df_data = df_data.append(tmp_data, ignore_index=True)

        # ======================= 将 DataFrame 转为 tensor =======================
        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        id = torch.tensor(df_data['id'].to_numpy())

        FTR_2_pred = torch.tensor(df_data['FTR_2_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())
        FTR_3_pred = torch.tensor(df_data['FTR_3_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())

        FTR_2_acc = torch.tensor(df_data['FTR_2_acc'].astype(int).to_numpy())
        FTR_3_acc = torch.tensor(df_data['FTR_3_acc'].astype(int).to_numpy())

        FTR_2 = df_data['FTR_2'].to_numpy()
        FTR_3 = df_data['FTR_3'].to_numpy()

        # 文本 -> BERT token_ids & masks
        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        
        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        # 这里的 TensorDataset 顺序要与 Trainer/data2gpu 中的 unpack 一一对应
        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            FTR_2_pred,
            FTR_2_acc,
            FTR_3_pred,
            FTR_3_acc,
            FTR_2_token_ids,
            FTR_2_masks,
            FTR_3_token_ids,
            FTR_3_masks,
            label,
            id,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()