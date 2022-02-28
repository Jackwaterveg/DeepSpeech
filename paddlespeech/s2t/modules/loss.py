# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from wenet(https://github.com/wenet-e2e/wenet)
import inspect

import paddle
from paddle import nn
from paddle.nn import functional as F

from paddlespeech.s2t.utils.log import Log

import os
import numpy as np
root_dir = "compare/result_store/paddlespeech"


logger = Log(__name__).getlog()

__all__ = ['CTCLoss', "LabelSmoothingLoss"]


class CTCLoss(nn.Layer):
    def __init__(self,
                 blank=0,
                 reduction='sum',
                 batch_average=False,
                 grad_norm_type=None):
        super().__init__()
        # last token id as blank id
        self.loss = nn.CTCLoss(blank=blank, reduction=reduction)
        self.batch_average = batch_average

        logger.info(
            f"CTCLoss Loss reduction: {reduction}, div-bs: {batch_average}")
        logger.info(f"CTCLoss Grad Norm Type: {grad_norm_type}")

        assert grad_norm_type in ('instance', 'batch', 'frame', None)
        self.norm_by_times = False
        self.norm_by_batchsize = False
        self.norm_by_total_logits_len = False
        if grad_norm_type is None:
            # no grad norm
            pass
        elif grad_norm_type == 'instance':
            self.norm_by_times = True
        elif grad_norm_type == 'batch':
            self.norm_by_batchsize = True
        elif grad_norm_type == 'frame':
            self.norm_by_total_logits_len = True
        else:
            raise ValueError(f"CTCLoss Grad Norm no support {grad_norm_type}")
        kwargs = {
            "norm_by_times": self.norm_by_times,
            "norm_by_batchsize": self.norm_by_batchsize,
            "norm_by_total_logits_len": self.norm_by_total_logits_len,
        }

        # Derive only the args which the func has
        try:
            param = inspect.signature(self.loss.forward).parameters
        except ValueError:
            # Some function, e.g. built-in function, are failed
            param = {}
        self._kwargs = {k: v for k, v in kwargs.items() if k in param}
        _notin = {k: v for k, v in kwargs.items() if k not in param}
        logger.info(f"{self.loss} kwargs:{self._kwargs}, not support: {_notin}")

    def forward(self, logits, ys_pad, hlens, ys_lens):
        """Compute CTC loss.

        Args:
            logits ([paddle.Tensor]): [B, Tmax, D]
            ys_pad ([paddle.Tensor]): [B, Tmax]
            hlens ([paddle.Tensor]): [B]
            ys_lens ([paddle.Tensor]): [B]

        Returns:
            [paddle.Tensor]: scalar. If reduction is 'none', then (N), where N = \text{batch size}.
        """
        B = paddle.shape(logits)[0]
        # warp-ctc need logits, and do softmax on logits by itself
        # warp-ctc need activation with shape [T, B, V + 1]
        # logits: (B, L, D) -> (L, B, D)
        logits = logits.transpose([1, 0, 2])
        ys_pad = ys_pad.astype(paddle.int32)
        loss = self.loss(logits, ys_pad, hlens, ys_lens, **self._kwargs)
        if self.batch_average:
            # Batch-size average
            loss = loss / B
        return loss


class LabelSmoothingLoss(nn.Layer):
    """Label-smoothing loss.
    In a standard CE loss, the label's data distribution is:
        [0,1,2] ->
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.
        e.g.
        smoothing=0.1
        [0,1,2] ->
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool=False):
        """Label-smoothing loss.

        Args:
            size (int): the number of class
            padding_idx (int): padding class id which will be ignored for loss
            smoothing (float): smoothing rate (0.0 means the conventional CE)
            normalize_length (bool):
                True, normalize loss by sequence length;
                False, normalize loss by batch size.
                Defaults to False.
        """
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.normalize_length = normalize_length
        self.criterion = nn.KLDivLoss(reduction="none")
        

        """
        padding_idx -1
        self.confidence 0.9
        self.smoothing 0.1
        self.size 4233

        """

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """Compute loss between x and target.
        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (paddle.Tensor): prediction (batch, seqlen, class)
            target (paddle.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (paddle.Tensor) : The KL loss, scalar float value
        """
        # print ("===========")
        # print ("padding_idx", self.padding_idx)
        # print ("self.confidence", self.confidence)
        # print ("self.smoothing",self.smoothing)
        # print ("self.size",self.size)
        # print ("self.normalize_length", self.normalize_length)
        """
        padding_idx -1
        self.confidence 0.9
        self.smoothing 0.1
        self.size 4233
        self.normalize_length False
        """
        B, T, D = paddle.shape(x)
        assert D == self.size
        x = x.reshape((-1, self.size))
        target = target.reshape([-1])

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = paddle.full_like(x, self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)

        #TODO(Hui Zhang): target = target * (1 - ignore)  # avoid -1 index
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        print ("target", target)
        # true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        target_mask = F.one_hot(target, self.size)
        true_dist *= (1 - target_mask)
        true_dist += target_mask * self.confidence
        print ("true_dist", true_dist)
        np.save(os.path.join(root_dir, "true_dist.npy"), true_dist.numpy())

        # print ("x", x)
        # print ("paddle softmax", F.softmax(x, axis=1))
        # print ("paddle.log_softmax", paddle.log_softmax(x, axis=1))
        # #print ("log_softmax",F.log_softmax(x, axis=1))
        # x_np = x.numpy()
        # import torch
        # x_torch_tensor = torch.tensor(x_np)
        # torch_softmax = torch.softmax(x_torch_tensor, dim=1)
        # torch_log_softmax = torch.log_softmax(x_torch_tensor, dim=1)
        # print (x_torch_tensor.size())
        # print ("torch softmax", torch_softmax)
        # print("torch_log_softmax", torch_log_softmax)
        # torch_log_softmax_np = torch_log_softmax.numpy()
        # torch_log_softmax_pdtensor = paddle.to_tensor(torch_log_softmax_np)

        #kl = self.criterion(paddle.log(F.softmax(x, axis=1)), true_dist
        kl = self.criterion(F.log_softmax(x, axis=1), true_dist)
        print ("kl", kl)
        log_softmax = F.log_softmax(x, axis=1)
        x_log_softmax_np = log_softmax.cpu().detach().numpy()
        np.save(os.path.join(root_dir, "log_softmax_.npy"), x_log_softmax_np)
       # kl = self.criterion(torch_log_softmax_pdtensor, true_dist)
        np.save("paddle_smoothing_x.npy", x.numpy())
        np.save("paddle_log_softmax.npy", F.log_softmax(x, axis=1).numpy())
        #TODO(Hui Zhang): sum not support bool type
        #total = len(target) - int(ignore.sum())
        total = len(target) - int(ignore.type_as(target).sum())
        print ("total", total)
        denom = total if self.normalize_length else B
        #numer = (kl * (1 - ignore)).sum()
        print ("ignore", ignore)
      #  numer = kl.masked_fill(ignore.unsqueeze(1), 0).sum() # ignore the impact of the ignored ids
        numer = kl.sum()
        print ("numer", numer)
        print ("denom", denom)
        res = numer / denom
        res_np = res.numpy()
        np.save(os.path.join(root_dir, "attn_res_.npy"), res_np)
        print ("attn_res_np", res_np)
        return res