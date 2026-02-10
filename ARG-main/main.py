import os
import argparse
import json
from utils.utils import get_tensorboard_writer

# 本文件是整个项目的入口：
# 1. 通过 argparse 解析命令行参数
# 2. 构造 config 字典（训练/模型的所有超参数）
# 3. 调用 grid_search.Run 来启动训练，并把最优结果保存到日志

parser = argparse.ArgumentParser()

# ======================= 基础训练/模型参数 =======================
parser.add_argument('--model_name', type=str, default='ARG')  # 选择使用的模型：ARG 或 ARG-D
parser.add_argument('--epoch', type=int, default=50)          # 最大训练轮数
parser.add_argument('--max_len', type=int, default=170)       # 每条样本的最大 token 长度
parser.add_argument('--early_stop', type=int, default=5)      # 早停轮数容忍度
parser.add_argument('--language', type=str, default='en')     # 语言（en / ch）
parser.add_argument('--root_path', type=str, default='D:\\复现代码\\ARG-main\\data\\ARG_dataset\\en')  # 数据集根路径
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3759)         # 随机种子
parser.add_argument('--gpu', type=str, default='0')           # 使用哪块 GPU
parser.add_argument('--emb_dim', type=int, default=768)       # BERT 输出维度
parser.add_argument('--co_attention_dim', type=int, default=300)
parser.add_argument('--lr', type=float, default=2e-4)         # 初始学习率
parser.add_argument('--save_log_dir', type=str, default= './logs')          # 日志保存目录
parser.add_argument('--save_param_dir', type=str, default= './param_model') # 模型权重保存目录
parser.add_argument('--param_log_dir', type=str, default = './logs/param')  # 记录超参数搜索结果

# ======================= 额外参数 =======================
parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorlog')  # TensorBoard 日志目录
parser.add_argument('--bert_path', type=str, default = 'D:\\复现代码\\models\\bert-base-uncased')      # 预训练 BERT 路径
parser.add_argument('--data_type', type=str, default = 'rationale')  # 数据类型，本项目只用 rationale
parser.add_argument('--data_name', type=str, default = 'train')      # 本次实验在日志/文件名中使用的标记
parser.add_argument('--eval_mode', type=bool, default = False)       # 是否仅评估已训练好的模型

# ======================= 模型结构控制相关 =======================
parser.add_argument('--expert_interaction_method', type=str, default = 'cross_attention')  # 专家间交互方式，当前实现用 cross_attention
parser.add_argument('--llm_judgment_predictor_weight', type=float, default = -1)          # LLM 判断预测器（simple_ftr_*）的损失权重
parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default = -1)  # rationale 有用性评估器（hard_ftr_*）的损失权重

# ======================= 蒸馏相关参数（ARG-D 使用） =======================
parser.add_argument('--kd_loss_weight', type=float, default=1)   # 学生/教师特征 MSE 蒸馏损失的权重
parser.add_argument('--teacher_path', type=str)                  # 预训练好教师模型（ARG）的权重路径

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 通过环境变量指定可见 GPU

from grid_search import Run
import torch
import numpy as np
import random

# ========== 设置随机种子，保证实验可复现 ==========
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};'.format \
    (args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))
print('data_type: {}; data_path: {}; data_name: {};'.format \
    (args.data_type, args.root_path, args.data_name))

# 把所有训练超参/路径等打包进一个 config 字典，后续 Trainer 统一读取
config = {
        'use_cuda': True,
        'seed': args.seed,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'language': args.language,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2},
            'llm_judgment_predictor_weight': args.llm_judgment_predictor_weight,
            'rationale_usefulness_evaluator_weight': args.rationale_usefulness_evaluator_weight,
            'kd_loss_weight': args.kd_loss_weight
            },
        'emb_dim': args.emb_dim,
        'co_attention_dim': args.co_attention_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir,

        'tensorboard_dir': args.tensorboard_dir,
        'bert_path': args.bert_path,
        'data_type': args.data_type,
        'data_name': args.data_name,
        'eval_mode': args.eval_mode,

        'teacher_path': args.teacher_path,  # 仅 ARG-D 会使用到的教师模型路径
        'month': 1                           # 目前固定只用第 1 个月份的数据
        }


if __name__ == '__main__':
    # 创建 TensorBoard writer，用于记录训练过程中的指标
    writer = get_tensorboard_writer(config)
    print('before in config')
    print(config)

    # 启动训练/搜索流程，返回最优指标（如 macro-F1）
    best_metric = Run(config = config, writer = writer).main()

    # 将最优结果保存为一个独立的 json，方便后续快速查看
    save_dir = './logs/log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, config['data_name']+'.json')
    with open(save_path, 'w') as file:
        json.dump(best_metric, file, indent=4, ensure_ascii=False)
