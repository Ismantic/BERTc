"""线性链 CRF。只依赖 torch,替掉 torchcrf。

MT 的 CWS head 和 NER head 各挂一个,是 joint score 1.4712 的组成部分,
所以接口和数值都跟 torchcrf 0.7.2 对齐(见 test/test_crf.py 的对拍):

  - 参数:start_transitions (T,)、end_transitions (T,)、transitions (T,T)
    其中 transitions[i, j] = 从 tag i 转到 tag j 的分数
  - 初始化:三者都是 uniform(-0.1, 0.1)
  - forward(emissions, tags, mask, reduction) 返回**对数似然**(不是 loss),
    调用方自己取负号
  - decode 返回 List[List[int]],每条只到该样本的真实长度

约定:mask 的第一个时间步必须全为 1(torchcrf 同款要求),因为序列起点
不能被 mask 掉;padding 只允许出现在尾部。
"""
from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """线性链 CRF。

    Args:
        num_tags: 标签数
        batch_first: True 时输入是 (B, L, T) / (B, L),否则 (L, B, T) / (L, B)
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"num_tags 必须为正: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    # ------------------------------------------------------------ 校验

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.Tensor] = None,
                  mask: Optional[torch.Tensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions 必须是 3 维,收到 {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(f"emissions 最后一维应为 {self.num_tags},"
                             f"收到 {emissions.size(2)}")
        if tags is not None and emissions.shape[:2] != tags.shape:
            raise ValueError(f"emissions 与 tags 前两维不一致: "
                             f"{tuple(emissions.shape[:2])} vs {tuple(tags.shape)}")
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f"emissions 与 mask 前两维不一致: "
                                 f"{tuple(emissions.shape[:2])} vs {tuple(mask.shape)}")
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask 的第一个时间步必须全为 1")

    # ------------------------------------------------------------ 对数似然

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = "sum") -> torch.Tensor:
        """返回给定标签序列的对数似然。

        reduction: none / sum / mean / token_mean
          none       → (B,) 每条一个值
          sum        → 标量,按 batch 求和
          mean       → 标量,按 batch 求平均(MT 训练用这个)
          token_mean → 标量,除以有效 token 数
        """
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"未知 reduction: {reduction}")
        self._validate(emissions, tags=tags, mask=mask)
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if mask.dtype != torch.uint8:
            mask = mask.to(torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 分子:真实路径分数;分母:所有路径的 logsumexp
        llh = self._score(emissions, tags, mask) - self._normalizer(emissions, mask)

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        return llh.sum() / mask.to(emissions.dtype).sum()

    def _score(self, emissions: torch.Tensor, tags: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        """真实标签路径的分数。emissions (L,B,T) / tags (L,B) / mask (L,B)。"""
        seq_len, batch_size = tags.shape
        mask = mask.to(emissions.dtype)

        # 起始转移 + 第 0 步发射
        score = self.start_transitions[tags[0]]
        score = score + emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            # 只在该步有效时累加,padding 步贡献 0
            score = score + self.transitions[tags[i - 1], tags[i]] * mask[i]
            score = score + emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 结束转移用每条序列最后一个有效位置的 tag
        last_idx = mask.long().sum(dim=0) - 1
        last_tags = tags[last_idx, torch.arange(batch_size)]
        return score + self.end_transitions[last_tags]

    def _normalizer(self, emissions: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """前向算法求配分函数的对数。"""
        seq_len = emissions.size(0)

        # (B, T):到第 0 步为止、以各 tag 结尾的路径分数
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            # (B, T, T):broadcast_score[b,i,j] = 到 i 结尾的分数
            #            broadcast_emis[b,i,j] = 第 i 步发射到 j 的分数
            broadcast_score = score.unsqueeze(2)
            broadcast_emis = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emis
            next_score = torch.logsumexp(next_score, dim=1)
            # padding 步保持原值
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        return torch.logsumexp(score + self.end_transitions, dim=1)

    # ------------------------------------------------------------ 解码

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Viterbi 解码,返回每条序列的最优标签路径。"""
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        if mask.dtype != torch.uint8:
            mask = mask.to(torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi(emissions, mask)

    def _viterbi(self, emissions: torch.Tensor,
                 mask: torch.Tensor) -> List[List[int]]:
        seq_len, batch_size = mask.shape

        score = self.start_transitions + emissions[0]   # (B, T)
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)            # (B, T, 1)
            broadcast_emis = emissions[i].unsqueeze(1)      # (B, 1, T)
            next_score = broadcast_score + self.transitions + broadcast_emis
            next_score, indices = next_score.max(dim=1)     # (B, T)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1

        best_paths = []
        for idx in range(batch_size):
            _, best_last = score[idx].max(dim=0)
            best = [best_last.item()]
            # 从该序列的真实终点往回回溯
            for hist in reversed(history[:seq_ends[idx]]):
                best.append(hist[idx][best[-1]].item())
            best.reverse()
            best_paths.append(best)
        return best_paths
