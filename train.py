import argparse
import logging
import os
import random
import math
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm.auto import tqdm, trange
from torch.autograd import Variable
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import IterableDataset, DataLoader, Dataset
import time
import torch.distributed as dist
import gc
from datetime import timedelta
from tokenizers import Tokenizer
import wandb

# 设置环境变量，禁用 Weights & Biases (wandb) 的模型监控以减少开销
os.environ["WANDB_WATCH"] = "false"

# 定义 BraLM（Brain Language Model）类，一个用于图序列建模的自定义神经网络
class BraLM(nn.Module):
    def __init__(self, hidden_size, use_ds=False, zero_freq_edges=None, vocab=None):
        """
        初始化 BraLM 模型。

        参数:
            hidden_size (int): 隐藏状态的维度。
            use_ds (bool): 是否使用 DeepSpeed 进行混合精度训练。
            zero_freq_edges (dict): 零频率边的字典，用于参数共享。
            vocab (Vocab): 包含节点和边映射的词汇表对象。
        """
        super().__init__()
        self.hidden_size = hidden_size  # 存储隐藏状态维度
        self.activation = nn.GELU()  # 使用 GELU 激活函数
        self.positions = nn.Parameter(torch.ones(1, 512, 1))  # 位置编码参数
        self.device = None  # 设备（CPU/GPU）将在后续设置
        
        # 为 Fully Sharded Data Parallel (FSDP) 兼容性准备
        self._tied_weights_keys = []  # 存储共享权重键的列表（用于参数共享）

        self.use_ds = use_ds  # DeepSpeed 使用标志
        self.zero_freq_edges = zero_freq_edges  # 零频率边的字典
        self.vocab = vocab  # 词汇表对象，用于节点/边映射

    def prepare_network(self, vocab):
        """
        根据词汇表初始化网络的权重和偏置参数。

        参数:
            vocab (Vocab): 包含边和节点映射的词汇表对象。
        """
        # 为扁平化结构创建索引映射
        self.weight_indices = {}  # 映射 (源索引, 目标索引) 到参数索引
        self.shared_param_idx = 0  # 零频率边的共享参数索引
        
        # 新参数的当前索引
        current_idx = 1
        
        # 遍历词汇表中的边，填充参数和映射
        for s_idx, s in enumerate(vocab.edge_dict):
            for t_idx, t in enumerate(vocab.edge_dict[s]):
                if self.zero_freq_edges is not None and t in self.zero_freq_edges[s]:
                    # 对零频率边使用共享参数
                    self.weight_indices[(s_idx, t_idx)] = self.shared_param_idx
                else:
                    # 为非零频率边分配唯一参数索引
                    self.weight_indices[(s_idx, t_idx)] = current_idx
                    current_idx += 1

        # 初始化权重和偏置参数，使用均匀分布随机值
        self.weights = nn.Parameter(torch.randn(current_idx, self.hidden_size, self.hidden_size).uniform_(-0.5, 0.5))
        self.biases = nn.Parameter(torch.randn(current_idx, 1, self.hidden_size).uniform_(-0.5, 0.5))
        self.node_bias = nn.Parameter(torch.randn(len(vocab.edge_dict), 1, self.hidden_size).uniform_(-0.5, 0.5))

    def to_device(self, device):
        """
        将模型参数移动到指定设备（CPU/GPU）。

        参数:
            device (torch.device): 目标设备。
        """
        self.weights.to(device)
        self.biases.to(device)
        self.positions.data = self.positions.data.to(device)
        self.device = device

    @staticmethod
    def _reshape12(x):
        """
        将张量的前两个维度合并。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 合并前两个维度后的张量。
        """
        return x.reshape(-1, x.size(-2), x.size(-1))
    
    def get_positional_encoding(self, seq_len, d_model):
        """
        生成序列的正弦位置编码。

        参数:
            seq_len (int): 序列长度。
            d_model (int): 模型维度（隐藏状态维度）。

        返回:
            torch.Tensor: 位置编码张量，形状为 (1, seq_len, d_model)。
        """
        position = torch.arange(0, seq_len).reshape(-1, 1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
        position_encoding = torch.zeros(seq_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding.unsqueeze(0).to(self.device)

    def get_initial_tensor(self, batch_size, d, pe):
        """
        为序列的第一个标记初始化能量张量。

        参数:
            batch_size (int): 批次大小。
            d (torch.Tensor): 包含边索引的输入张量。
            pe (torch.Tensor): 位置编码张量。

        返回:
            torch.Tensor: 初始化后的能量张量，形状为 (batch_size, 1, hidden_size)。
        """
        # 初始化能量张量为全1，并按隐藏维度缩放
        energy_tensor = torch.ones(batch_size, 1, self.hidden_size) / self.hidden_size
        energy_tensor = energy_tensor.to(self.device)

        # 添加节点偏置和位置编码，然后应用激活函数
        node_bias = self.node_bias[d[:, 0, 0]]
        energy_tensor = self.activation(energy_tensor + node_bias + Variable(pe[:,0], requires_grad=False))
        return energy_tensor

    def forward(self, neighbor_ids):
        """
        模型的前向传播，计算损失。

        参数:
            neighbor_ids (torch.Tensor): 形状为 (batch_size, seq_len, 1+num_neg_samples, 2) 的张量，
                                        包含正样本和负样本的边索引。

        返回:
            torch.Tensor: 序列的平均损失。
        """
        batch_size = neighbor_ids.size(0)
        loss = 0

        # 生成位置编码
        pe = self.get_positional_encoding(512, self.hidden_size)

        # 按序列中的每个标记进行处理
        for i in range(neighbor_ids.size(1)):
            d = neighbor_ids[:, i]  # (batch_size, 1+num_neg_samples, 2)
            
            # 为第一个标记初始化能量张量
            if i == 0:
                energy_tensor = self.get_initial_tensor(batch_size, d, pe)
            else:
                # 使用位置权重聚合之前的能量张量
                energy_tensor = (energy_cache * self.positions[:, :i, :].softmax(1)).sum(1, keepdim=True)

            # 向量化参数查找权重和偏置
            src_idx = d[..., 0]  # 源索引
            tgt_idx = d[..., 1]  # 目标索引
            param_indices = torch.tensor([self.weight_indices.get((s.item(), t.item()), self.shared_param_idx) 
                                        for s, t in zip(src_idx.reshape(-1), tgt_idx.reshape(-1))], 
                                       device=self.device).reshape(batch_size, -1)
            
            # 批量获取所有边的权重和偏置
            w = self.weights[param_indices]
            b = self.biases[param_indices]

            # 计算下一个能量张量
            expand_energy_tensor = self._reshape12(energy_tensor.unsqueeze(1).repeat(1, w.size(1), 1, 1))
            if self.use_ds:
                expand_energy_tensor = expand_energy_tensor.half()  # 为 DeepSpeed 转换为 FP16
            nxt_energy_tensor = self.activation(expand_energy_tensor.bmm(self._reshape12(w)) + self._reshape12(b) + 
                                              Variable(pe[:,i+1], requires_grad=False))
            output_tensor = nxt_energy_tensor.reshape(batch_size, -1, nxt_energy_tensor.size(-2), nxt_energy_tensor.size(-1))

            # 更新能量缓存
            if i == 0:
                energy_cache = output_tensor[:,0]
            else:
                energy_cache = torch.cat([energy_cache, output_tensor[:,0]], dim=1)

            # 计算交叉熵损失
            energy = output_tensor.norm(2, (-2, -1))
            label = torch.LongTensor([0 for _ in range(batch_size)]).to(self.device)
            loss += nn.CrossEntropyLoss()(energy, label)

        return loss / neighbor_ids.size(1)  # 返回序列长度的平均损失

    def decode(self, start, vocab, max_new_tokens=16, do_sample=False, temperature=1):
        """
        从初始边序列解码生成新序列。

        参数:
            start (list): 初始边序列（边元组列表）。
            vocab (Vocab): 词汇表对象，用于节点/边映射。
            max_new_tokens (int): 生成新标记的最大数量。
            do_sample (bool): 是否从概率分布中采样（否则取最大值）。
            temperature (float): 采样温度，控制随机性。

        返回:
            list: 生成的边序列（边元组列表）。
        """
        ret = []
        pe = self.get_positional_encoding(512, self.hidden_size)
        
        # 处理初始序列
        for i, pair in enumerate(start):
            if i == 0:
                energy_tensor = self.get_initial_tensor(batch_size=1, d=torch.tensor([[pair]], device=self.device), pe=pe).squeeze(0)
            else:
                energy_tensor = (energy_cache * self.positions[:, :i, :].softmax(1)).sum(1, keepdim=True).squeeze(0)
            
            # 获取当前边的权重和偏置
            param_idx = self.weight_indices.get((pair[0], pair[1]), self.shared_param_idx)
            w = self.weights[param_idx].to(self.device)
            b = self.biases[param_idx].to(self.device)

            # 计算下一个能量张量
            energy_tensor = self.activation(energy_tensor.mm(w) + b + pe.squeeze(0)[i])
            if i == 0:
                energy_cache = energy_tensor.unsqueeze(0)
            else:
                energy_cache = torch.cat([energy_cache, energy_tensor.unsqueeze(0)], dim=1)
            ret += [pair]
        
        x = pair[1]  # 初始序列的最后一个节点
        prev_i = len(start)

        # 生成新标记
        for i in range(max_new_tokens):
            # 获取当前节点的候选边
            candidates = vocab(vocab.get_neighbor_of_node(x, -1))
            
            # 获取所有候选边的权重和偏置
            param_indices = torch.tensor([self.weight_indices.get((x, t[1]), self.shared_param_idx) 
                                        for t in candidates], device=self.device)
            all_w = self.weights[param_indices].to(self.device)
            all_b = self.biases[param_indices].to(self.device)

            curr_i = prev_i + i
            energy_tensor = (energy_cache * self.positions[:, :curr_i, :].softmax(1)).sum(1, keepdim=True)
            expand_energy_tensor = energy_tensor.unsqueeze(1).repeat(1, all_w.size(0), 1, 1)
            expand_energy_tensor = self._reshape12(expand_energy_tensor)

            # 计算所有候选边的能量
            nxt_energy_tensor = self.activation(expand_energy_tensor.bmm(self._reshape12(all_w)) + self._reshape12(all_b) + 
                                               pe[:,curr_i].unsqueeze(0))
            output_tensor = nxt_energy_tensor.reshape(1, -1, nxt_energy_tensor.size(-2), nxt_energy_tensor.size(-1))

            # 计算概率
            energy = output_tensor.norm(2, (-2,-1)).squeeze()
            probs = torch.softmax(energy, dim=-1)
            if temperature > 0:
                probs = probs / temperature
            if do_sample:
                index = torch.multinomial(probs, 1).item()  # 从概率分布中采样
            else:
                index = probs.argmax(-1).item()  # 取最大概率的候选边

            y = candidates[index][-1]
            ret += [(x, y)]

            energy_tensor = output_tensor[0, index]
            x = y
            energy_cache = torch.cat([energy_cache, energy_tensor.unsqueeze(0)], dim=1)

        return ret

# 定义 Vocab 类，用于管理节点和边映射
class Vocab:
    def __init__(self, node_dict, nodeindex_dict, edge_dict, edge_decode_dict):
        """
        初始化词汇表对象。

        参数:
            node_dict (dict): 节点名称到索引的映射。
            nodeindex_dict (dict): 索引到节点名称的映射。
            edge_dict (dict): 源节点到目标节点及其索引的映射。
            edge_decode_dict (dict): 边索引到边字符串的映射。
        """
        self.node_dict = node_dict  # {节点: 索引}
        self.nodeindex_dict = nodeindex_dict  # {索引: 节点}
        self.edge_dict = edge_dict  # {节点: {目标节点: (源索引, 目标索引)}}
        self.edge_decode_dict = edge_decode_dict  # {(源索引, 目标索引): 边字符串}

    def __call__(self, x):
        """
        将输入转换为边索引。

        参数:
            x (str or list): 边字符串或边字符串列表。

        返回:
            list or tuple: 边索引。
        """
        if isinstance(x, list):
            return [self.__call__(_) for _ in x]
        else:
            return self.fetch(x)

    def fetch(self, x):
        """
        为给定的边字符串获取边索引。

        参数:
            x (str): 边字符串，格式为 "源->目标"。

        返回:
            tuple: 边索引 (源索引, 目标索引)。
        """
        s, t = x.split("->")
        return self.edge_dict[s][t] if s in self.edge_dict and t in self.edge_dict[s] else self.edge_dict[""][""]

    @classmethod
    def from_node_dict(cls, dictname):
        """
        从节点字典创建 Vocab 实例。

        参数:
            dictname (dict): 节点名称到索引的映射字典。

        返回:
            Vocab: 初始化后的 Vocab 对象。
        """
        node_dict = dict()
        nodeindex_dict = dict()
        edge_dict = dict()
        edge_decode_dict = dict()
        for s in dictname:
            node_dict[s] = dictname[s]
            nodeindex_dict[dictname[s]] = s
            edge_dict[s] = {}
            for t in dictname:
                edge_dict[s][t] = (dictname[s], dictname[t])
                edge_decode_dict[(dictname[s], dictname[t])] = "->".join([s, t])
        return cls(node_dict, nodeindex_dict, edge_dict, edge_decode_dict)

    @classmethod
    def from_edge(cls, filename):
        """
        从边文件创建 Vocab 实例。

        参数:
            filename (str): 边文件路径。

        返回:
            Vocab: 初始化后的 Vocab 对象。
        """
        edge_dict = dict()
        edge_dict[""] = {}
        edge_dict[""][""] = (0, 0)
        edge_decode_dict = dict()
        with open(filename) as f:
            for line in f:
                s, t = line.strip().split("->")
                if s not in edge_dict:
                    i = len(edge_dict)
                    j = 0
                    edge_dict[s] = dict()
                else:
                    i = edge_dict[s][list(edge_dict[s].keys())[0]][0]
                    j = len(edge_dict[s])
                edge_dict[s][t] = (i, j)
                edge_decode_dict[(i, j)] = "->".join([s, t])
        return cls(None, edge_dict, edge_decode_dict)

    def get_neighbor_of_edge(self, key, k, frequency_dict=None):
        """
        为一个边获取 k 个负样本。

        参数:
            key (str): 边字符串，格式为 "源->目标"。
            k (int): 返回的负样本数量。
            frequency_dict (dict): 用于采样的词频字典。

        返回:
            list: k 个负样本边字符串的列表。
        """
        s, t = key.split("->")
        _s = s if s in self.edge_dict else ""
        
        if frequency_dict:
            frequency_lst = list(frequency_dict[_s].keys())
            t_lst = [x for x in frequency_lst[:k+1] if x != t][:k]
            ret = ["->".join([_s, _t]) for _t in t_lst]
            random.shuffle(ret)
            return ret
        else:
            ret = ["->".join([_s, _t]) for _t in self.edge_dict[_s].keys() if _t != t]
            random.shuffle(ret)
            return ret[:k] if k != -1 else ret

    def get_neighbor_of_node(self, key, k):
        """
        为给定的节点索引获取 k 个邻居边。

        参数:
            key (int): 节点索引。
            k (int): 返回的邻居边数量。

        返回:
            list: k 个邻居边字符串的列表。
        """
        s = self.nodeindex_dict[key]
        ret = ["->".join([s, _t]) for _t in self.edge_dict[s].keys() if _t != s]
        random.shuffle(ret)
        return ret[:k] if k != -1 else ret
    
    def get_neighbor_of_edge_broadcast(self, key, edges, k=100):
        """
        为一组边获取 k 个负样本。

        参数:
            key (str): 参考边字符串。
            edges (list): 边字符串列表。
            k (int): 每个边的负样本数量。

        返回:
            list: 包含每个边 k 个负样本的列表。
        """
        s, t = key.split("->")
        _ret = [_t for _t in self.edge_dict[s].keys() if _t != t]
        random.shuffle(_ret)
        ret = []
        for edge in edges:
            s, t = edge.split("->")
            ret += [["->".join([s, _t]) for _t in _ret[:k]]] 
        return ret

    @staticmethod
    def to_path(tokens):
        """
        将标记列表转换为边字符串列表。

        参数:
            tokens (list): 节点标记列表。

        返回:
            list: 边字符串列表，格式为 "源->目标"。
        """
        path = []
        for left, right in zip(tokens[:-1], tokens[1:]):
            path.append("->".join([left, right]))
        return path

    def get_edge_of_node(self, key):
        """
        获取给定节点的所有边。

        参数:
            key (str): 节点名称。

        返回:
            list: 边索引元组的列表。
        """
        return list(self.edge_dict[key].values())

    def decode(self, x):
        """
        将边索引元组解码为字符串表示。

        参数:
            x (tuple): 边索引元组 (源索引, 目标索引)。

        返回:
            str: 边字符串，格式为 "源->目标"。
        """
        return self.edge_decode_dict[x]

# 配置日志记录，用于调试和监控
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 实用函数，将全角字符标准化为半角 ASCII 字符
def stdf(string):
    """
    将全角字符转换为半角 ASCII 字符。

    参数:
        string (str): 输入字符串。

    返回:
        str: 标准化的 ASCII 字符串。
    """
    def _h(char):
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020  # 将全角空格转换为普通空格
        else:
            inside_code -= 0xfee0  # 将全角字符转换为半角
        if inside_code < 0x0020 or inside_code > 0x7e:
            return char
        return chr(inside_code)
    return "".join([_h(char) for char in string])

# 定义 WikiDataset 类，用于加载和处理维基百科数据
class WikiDataset(Dataset):
    """
    用于训练的维基百科数据处理类。
    """
    def __init__(self, filename, vocab, max_seq_length, num_neg_samples, seed, buffer_size=100000, shuffle=True, use_frequency=False, use_bpe=False, bpe_tokenizer=None):
        """
        初始化数据集。

        参数:
            filename (str): 数据文件路径。
            vocab (Vocab): 词汇表对象。
            max_seq_length (int): 最大序列长度。
            num_neg_samples (int): 每个边的负样本数量。
            seed (int): 随机种子，确保可重复性。
            buffer_size (int): 数据加载缓冲区大小（本实现未使用）。
            shuffle (bool): 是否打乱数据。
            use_frequency (bool): 是否使用词频进行负采样。
            use_bpe (bool): 是否对英文使用 Byte-Pair Encoding (BPE) 分词。
            bpe_tokenizer (dict): BPE 分词器配置。
        """
        super().__init__()
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.num_neg_samples = num_neg_samples
        self.generator = np.random.default_rng(seed=seed)
        self.use_bpe = use_bpe
        self.bpe_tokenizer = bpe_tokenizer
        
        self.data = self.read(filename)
        
        if use_frequency:
            freq_file = 'word_frequency_en.json' if use_bpe else 'word_frequency.json'
            with open(freq_file, 'r') as f:
                self.frequency_dict = json.load(f)
        else:
            self.frequency_dict = None

    def read(self, filename):
        """
        从输入文件读取数据。

        参数:
            filename (str): 数据文件路径。

        返回:
            list: 处理后的行数据（标记或原始文本）。
        """
        lines = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if self.use_bpe:
                    lines.append(line.strip())
                else:
                    src = list(line.strip()[:self.max_seq_length])
                    lines.append(src)
        return lines

    def __len__(self):
        """
        返回数据集中的样本数量。

        返回:
            int: 数据集大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。

        参数:
            idx (int): 样本索引。

        返回:
            torch.Tensor: 包含正样本和负样本的边索引张量。
        """
        src = self.data[idx]
        return self.vectorize(src)

    def vectorize(self, src):
        """
        将源序列转换为包含负样本的边索引。

        参数:
            src (str or list): 源序列（文本或标记列表）。

        返回:
            torch.Tensor: 形状为 (seq_len, 1+num_neg_samples, 2) 的张量，包含边索引。
        """
        if self.use_bpe:
            # 使用 BPE 分词
            bpe_tokens = self.bpe_tokenizer.encode(src).tokens
            pad_token = "[PAD]"
            if len(bpe_tokens) > self.max_seq_length:
                bpe_tokens = bpe_tokens[:self.max_seq_length]
            else:
                bpe_tokens.extend(pad_token for _ in range(self.max_seq_length - len(bpe_tokens)))
            tokens = bpe_tokens
        else:
            # 填充或截断序列
            if len(src) > self.max_seq_length:
                src = src[:self.max_seq_length]
            else:
                src.extend("" for _ in range(self.max_seq_length-len(src)))
            tokens = src
            
        # 将标记转换为边字符串
        edges = self.vocab.to_path(tokens)
        edge_ids = self.vocab(edges)
        edge_ids = edge_ids[:self.max_seq_length]
        # 为每个边获取负样本
        neighbor_ids = [self.vocab(self.vocab.get_neighbor_of_edge(e, self.num_neg_samples, self.frequency_dict)) for e in edges]

        # 组合正样本和负样本
        new_neighbor_ids = []
        for i, e_ids in enumerate(edge_ids):
            new_neighbor_ids.append([e_ids] + neighbor_ids[i])
        return torch.LongTensor(new_neighbor_ids)

# 主函数，负责协调训练和评估
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/wiki",
                        help="输入数据文件所在目录。")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="保存模型检查点和预测的目录。")
    parser.add_argument("--load_state_dict", type=str, default=None,
                        help="用于评估的已训练模型权重路径。")
    parser.add_argument("--do_train", action="store_true",
                        help="是否执行训练。")
    parser.add_argument("--do_eval", action="store_true",
                        help="是否在验证集上进行评估。")
    parser.add_argument("--num_neg_samples", type=int, default=100,
                        help="每个边的负样本数量。")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="分词后的最大序列长度。")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="训练批次大小。")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="评估批次大小。")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Adam 优化器的初始学习率。")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="训练的总轮数。")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="训练的最大步数（若设置，则覆盖轮数）。")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="训练的 L2 权重衰减。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数，累积后执行一次反向传播。")
    parser.add_argument("--no_cuda", action="store_true",
                        help="即使 CUDA 可用也禁用。")
    parser.add_argument("--fp16", action="store_true",
                        help="使用混合精度训练。")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，确保可重复性。")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每多少步保存一次检查点。")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="模型的隐藏状态维度。")
    parser.add_argument("--local_rank", type=int,
                        help="分布式训练的本地排名。")
    parser.add_argument("--initial_file_number", type=int, default=0,
                        help="训练数据的起始文件编号。")
    parser.add_argument("--end_file_number", type=int, default=0,
                        help="训练数据的结束文件编号。")
    parser.add_argument("--wiki_sorted_size", type=int, default=70,
                        help="排序后的维基数据文件总数。")
    parser.add_argument("--run_name", type=str, default="plusb_pluspe_order",
                        help="wandb 运行名称。")
    parser.add_argument("--use_frequency", action="store_true",
                        help="是否使用词频进行负采样。")
    parser.add_argument("--train_full", type=str, default=None,
                        help="在单个完整文本文件上训练的路径。")
    parser.add_argument("--checkpoint_save_step", type=int, default=0,
                        help="保存检查点的步数间隔（仅在 train_full 为 True 时支持）。")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="恢复训练的检查点路径。")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="数据加载的工作进程数。")
    parser.add_argument("--vocab_path", type=str, default="vocab_wiki_4k.json",
                        help="词汇表文件路径。")
    parser.add_argument("--use_ds", action="store_true",
                        help="是否使用 DeepSpeed 进行训练。")
    parser.add_argument("--sparse", action="store_true",
                        help="是否为零频率边使用稀疏参数共享。")
    parser.add_argument("--use_bpe", action="store_true",
                        help="是否对英文使用 Byte-Pair Encoding 分词。")
    parser.add_argument("--bpe_tokenizer_path", type=str, default="wiki_bpe_tokenizer_4000_bytelevel.json",
                        help="BPE 分词器文件路径。")
    
    args = parser.parse_args()
    
    # 设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("设备: {}, GPU 数量: {}, 分布式训练: {}, 16位训练: {}".format(
        device, n_gpu, "-accelerate", args.fp16))

    # 根据梯度累积调整批次大小
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 从文件加载词汇表
    with open(args.vocab_path) as f:
        node_dict = json.load(f)
    vocab = Vocab.from_node_dict(node_dict)

    # 为稀疏参数共享加载零频率边
    if args.sparse:
        with open('word_frequency.json', 'r') as f:
            freq_dict = json.load(f)
        zero_freq_edges = {}
        for s in freq_dict:
            zero_freq_edges[s] = []
            for t in freq_dict[s]:
                if freq_dict[s][t] == 0:
                    zero_freq_edges[s].append(t)
    else:
        zero_freq_edges = None

    def stat_cuda(epoch, cur_file_num, step, location):
        """
        将 CUDA 内存统计信息记录到文件。

        参数:
            epoch (int): 当前轮数。
            cur_file_num (str or int): 当前文件编号。
            step (int): 当前训练步数。
            location (str): 记录统计信息的位置。
        """
        if accelerator.is_local_main_process:
            with open("cuda_stat.txt", "a") as f:
                if epoch is not None:
                    f.write('轮数: %d, 当前文件编号: %d, 步数: %d\n' % (epoch, cur_file_num, step))
                f.write(f'--{location}\n')
                f.write('已分配: %dG, 最大分配: %dG, 缓存: %dG, 最大缓存: %dG\n' % (
                    torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
                ))

    if args.do_train:
        # 配置分布式训练
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1080000))
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs], cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        # 初始化模型
        model = BraLM(args.hidden_size, args.use_ds, zero_freq_edges, vocab=vocab)
        model.prepare_network(vocab)

        # 如果指定，从检查点加载模型权重
        if args.load_state_dict:
            print(f"从检查点加载模型: {args.load_state_dict}")
            checkpoint = torch.load(args.load_state_dict, map_location="cpu")
            model.load_old(checkpoint["model_state_dict"])

        # 如果指定，从检查点恢复训练
        wandb_id = None
        global_step = 0
        if args.resume_from_checkpoint:
            print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint.get("global_step", 0)
            wandb_id = checkpoint.get("wandb_id")
        else:
            start_epoch = 0

        model.to_device(device)

        # 初始化优化器
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        if args.resume_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 为分布式训练准备模型和优化器
        if not args.use_ds:
            model, optimizer = accelerator.prepare(model, optimizer)

        if args.do_train:
            if accelerator.is_local_main_process:
                # 初始化 wandb 用于日志记录
                wandb.init(
                    project="brain",
                    name=args.run_name,
                    id=wandb_id,
                    resume="allow",
                    config=vars(args)
                )
                wandb.define_metric("custom_step")
                wandb.define_metric("batch_*", step_metric="custom_step")
                wandb.define_metric("epoch")
                wandb.define_metric("epoch_*", step_metric="epoch")
                print(f"启动 wandb 运行，ID: {wandb.run.id}")
                print(f"查看运行: {wandb.run.get_url()}")
            
            # 准备数据集和数据加载器
            if args.train_full:
                cur_file_num = args.train_full
                cur_filename = f"{cur_file_num}.txt"
                if args.use_bpe:
                    with open(args.bpe_tokenizer_path, 'r') as f:
                        bpe_tokenizer = json.load(f)
                else:
                    bpe_tokenizer = None
                dataset = WikiDataset(
                    os.path.join(args.data_dir, cur_filename), 
                    vocab, 
                    args.max_seq_length, 
                    args.num_neg_samples, 
                    seed=args.seed, 
                    shuffle=True, 
                    use_frequency=args.use_frequency,
                    use_bpe=args.use_bpe,
                    bpe_tokenizer=bpe_tokenizer
                )
                train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
                train_dataloader = accelerator.prepare(train_dataloader)
            elif args.resume_from_checkpoint:
                cur_file_num = checkpoint["cur_file_num"]
                if isinstance(cur_file_num, int) or cur_file_num.isdigit():
                    cur_file_num = int(cur_file_num) + 1
            else:
                cur_file_num = args.initial_file_number

            # 调整恢复训练的起始轮数和文件编号
            if args.resume_from_checkpoint and global_step > 0:
                if args.train_full and global_step % len(train_dataloader) == 0:
                    start_epoch = start_epoch + 1
                if not args.train_full and cur_file_num > args.end_file_number:
                    start_epoch = start_epoch + 1
                    cur_file_num = args.initial_file_number

            # 训练循环
            for epoch in trange(start_epoch, int(args.num_train_epochs), desc="轮次"):
                if epoch != start_epoch or args.train_full:
                    cur_file_num = args.initial_file_number
                while cur_file_num <= args.wiki_sorted_size:
                    if args.train_full:
                        cur_file_num = args.train_full
                    logger.info("***** 为 wiki = %s 执行训练 *****", cur_file_num)
                    logger.info("  批次大小 = %d", args.train_batch_size * accelerator.num_processes)
                    
                    # 加载当前文件的数据
                    if not args.train_full:
                        cur_filename = f"{cur_file_num}.txt"
                        if args.use_bpe:
                            with open(args.bpe_tokenizer_path, 'r') as f:
                                bpe_tokenizer = json.load(f)
                        else:
                            bpe_tokenizer = None
                        dataset = WikiDataset(
                            os.path.join(args.data_dir, cur_filename), 
                            vocab, 
                            args.max_seq_length, 
                            args.num_neg_samples, 
                            seed=args.seed, 
                            shuffle=True, 
                            use_frequency=args.use_frequency,
                            use_bpe=args.use_bpe,
                            bpe_tokenizer=bpe_tokenizer
                        )
                        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
                        if not args.use_ds:
                            train_dataloader = accelerator.prepare(train_dataloader)
                        else:
                            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

                    # 单个文件的训练循环
                    train_loss = 0
                    num_train_examples = 0
                    if accelerator.is_local_main_process:
                        progress_bar = tqdm(train_dataloader, desc="迭代")
                    
                    for step, batch in enumerate(train_dataloader, start=global_step % len(train_dataloader)):
                        batch_train_loss = 0
                        batch_num_train_examples = 0
                        # 仅对最后一个标记进行训练
                        for ind in range(batch.size(1) - 1, batch.size(1)):
                            model.train()
                            neighbor_ids = batch[:, :ind]
                            outputs = model(neighbor_ids)
                            loss = outputs

                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps
                            accelerator.backward(loss)

                            if n_gpu > 1:
                                dist.all_reduce(loss)
                                loss = loss / dist.get_world_size()

                            train_loss += loss.detach().item()
                            batch_train_loss += loss.detach().item()
                            num_train_examples += 1
                            batch_num_train_examples += 1
                            
                            del outputs
                            del loss
                            del neighbor_ids
                            gc.collect()

                            if (step + 1) % args.gradient_accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()

                                ppl = math.exp(batch_train_loss / batch_num_train_examples)
                                if accelerator.is_local_main_process:
                                    progress_bar.update(1)
                                    progress_bar.set_postfix(loss=batch_train_loss / batch_num_train_examples, perplexity=ppl)
                                    wandb.log({
                                        "batch_loss": batch_train_loss / batch_num_train_examples, 
                                        "batch_perplexity": math.exp(batch_train_loss / batch_num_train_examples),
                                        "batch_epoch": epoch,
                                        "custom_step": global_step
                                    })

                        global_step += 1

                        # 按指定间隔保存检查点
                        if accelerator.is_local_main_process and args.checkpoint_save_step > 0 and global_step % args.checkpoint_save_step == 0:
                            output_dir_f = f"{args.output_dir}/HS{args.hidden_size}/step_{global_step}/"
                            if not os.path.exists(output_dir_f):
                                os.makedirs(output_dir_f)
                            output_model_file = os.path.join(output_dir_f, f"checkpoint_{global_step}.bin")
                            model_to_save = model.module if hasattr(model, "module") else model
                            checkpoint = {
                                "model_state_dict": model_to_save.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), 
                                "epoch": epoch,
                                "global_step": global_step,
                                "args": vars(args),
                                "wandb_id": wandb.run.id
                            }
                            if not args.train_full:
                                checkpoint["cur_file_num"] = cur_file_num
                            print(f"保存检查点到 {output_model_file}")
                            torch.save(checkpoint, output_model_file)
                            print(f"检查点已保存到 {output_model_file}")

                    # 保存当前训练文件的模型
                    if accelerator.is_local_main_process:
                        epoch_avg_loss = train_loss / num_train_examples
                        epoch_ppl = math.exp(epoch_avg_loss)
                        wandb.log({
                            "epoch_loss": epoch_avg_loss,
                            "epoch_perplexity": epoch_ppl,
                            "epoch": epoch,
                        })
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_dir_f = f"{args.output_dir}/HS{args.hidden_size}/EPOCH{epoch}/"
                        if not os.path.exists(output_dir_f):
                            os.makedirs(output_dir_f)
                        output_model_file = os.path.join(output_dir_f, "f{}_pytorch_model.bin".format(cur_file_num))
                        if args.train_full or cur_file_num == args.end_file_number:
                            checkpoint = {
                                "model_state_dict": model_to_save.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch,
                                "global_step": global_step,
                                "args": vars(args),
                                "wandb_id": wandb.run.id
                            }
                            if not args.train_full:
                                checkpoint["cur_file_num"] = cur_file_num
                            print(f"保存模型到 {output_model_file}")
                            torch.save(checkpoint, output_model_file)
                            print(f"模型已保存到 {output_model_file}")

                    if args.train_full:
                        break
                    cur_file_num += 1
                    if cur_file_num > args.end_file_number:
                        break

if __name__ == "__main__":
    main()
