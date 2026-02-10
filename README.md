
#### 1. 代码结构

- **入口与调度**
  - `main.py`：解析命令行参数，构造统一的 `config`，设置随机种子与 GPU，并调用 `grid_search.Run(config, writer).main()` 启动训练与测试。
  - `grid_search.py`：根据 `config['model_name']` 选择 `ARG` 或 `ARG-D` 的 `Trainer`，目前以固定学习率形式组织为“网格搜索”结构，并记录最优指标与模型路径。

- **模型与训练**
  - `arg.py`：实现 ARG 主模型 `ARGModel` 及其 `Trainer`，包含多源信息融合、主任务与辅助任务训练、验证/测试以及早停逻辑。
  - `argd.py`：实现蒸馏模型 `ARGDModel` 及其 `Trainer`，利用已训练好的 ARG 作为教师进行特征蒸馏，仅用正文完成假新闻检测。

- **基础组件与数据**
  - `layers.py`：提供通用网络模块，如 `MLP`、`MaskAttention`、`SelfAttentionFeatureExtract`、`ParallelCoAttentionNetwork` 等注意力与特征提取层。
  - `dataloader.py`：从 JSON 文件读取样本，构造新闻正文、LLM rationale 及其预测/正确性标签，使用 `BertTokenizer` 编码并封装成 `DataLoader`。
  - `utils.py`：包含 `data2gpu`、`metrics`、`Recorder`、`get_monthly_path`、`get_tensorboard_writer` 等训练/日志辅助工具函数。
#### 2. ARG 模型流程

ARG 读取新闻正文和两路 LLM rationale，分别用 BERT 编码后，通过 cross-attention 建立正文与 rationale 的交互，得到两个特征并按得分重加权，再与正文特征一起做注意力聚合，经 MLP 输出假新闻概率；训练时同时加入“rationale 是否有用”和“LLM 预测标签”两个辅助任务，最终在验证集上根据指标早停，并在测试集上评估性能。

#### 3. ARG-D 模型流程

ARG-D 先加载训练好的 ARG 作为教师模型，训练阶段学生模型仅使用新闻正文，经 BERT + Transformer Block + 注意力 + MLP 得到预测结果，一方面用 BCE 拟合真实标签，另一方面用 MSE 拟合教师的最终特征表示，两者加权作为总损失进行联合优化，得到的学生模型在推理时只依赖新闻正文即可完成假新闻检测。
