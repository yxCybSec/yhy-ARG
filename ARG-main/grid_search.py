import logging
import os
import json
import random
import torch
import numpy as np

# 本文件负责根据 config 启动具体的 Trainer，并（可选地）做简单的超参数搜索。
# 目前只对学习率 lr 做“单点搜索”（列表里只有一个值），但结构上可以扩展为网格搜索。

from models.arg import Trainer as ARGTrainer
from models.argd import Trainer as ARGDTrainer

def setup_seed(seed):
    """为了在不同实验/不同进程中复现实验结果，统一设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    """浮点数范围生成器，目前并未真实使用到（可以用于 lr 连续搜索）。"""
    while x < y:
        x = round(x, 8)
        yield x
        x += jump

class Run():
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        """创建一个写入到指定 log_file 的 logger（只创建一次 handler）。"""
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(level = logging.INFO)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        """将 configinfo 转为字典的工具函数（当前未被使用）。"""
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        """
        训练主入口：
        1. 为当前模型+数据集构造参数日志文件；
        2. 遍历待搜索的超参数组合（这里只有 lr 一项）；
        3. 根据 model_name 创建对应的 Trainer（ARG / ARG-D），调用其 train()；
        4. 记录每次实验结果到 json，并挑选 metric 最优的一组参数。
        """
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)

        # 可以在这里扩展其他超参数搜索，例如 'batchsize'、'kd_loss_weight' 等
        train_param = { 'lr': [self.config['lr']] }

        print(train_param)
        param = train_param
        best_param = []

        json_dir = os.path.join(
            './logs/json/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        json_path = os.path.join(
            json_dir,
            'month_' + str(self.config['month']) + '.json'
        )
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # 保存每一个（超参设置 -> 训练结果）的列表，后面会写入到 json 文件
        json_result = []
        for p, vs in param.items():
            setup_seed(self.config['seed'])
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v

                if self.config['model_name'] == 'ARG':
                    trainer = ARGTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'ARG-D':
                    trainer = ARGDTrainer(self.config, self.writer)
                else:
                    raise ValueError('model_name is not supported')

                metrics, model_path, train_epochs = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': metrics,
                    'train_epochs': train_epochs,
                })

                if metrics['metric'] > best_metric['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric
