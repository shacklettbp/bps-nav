# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Petuum, Inc.  nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import functools
from typing import Any, Dict, Optional
import copy

import numpy as np
from torch.autograd import Variable
import torch.distributed


@torch.jit.script
def _sum_squared_mul(x, alpha: float):
    z = x * alpha
    return (z * z).sum()


class FP16AdaScale(object):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. code-block:: python

        optim = torch.optim.SGD(model, lr=0.001)
        model = DistributedDataParallel(model)
        adascale = AdaScale(optim)

        for epoch in ...:
            for batch in ...:
                optim.zero_grad()
                loss = ...
                loss.backward()
                adascale.step()

    Arguments:
        optimizer (torch.optim.Optimizer): Optimizer to apply AdaScale to.
        world_size (int): Number of world_size for distributed training. If
            None, defaults to ``torch.distributed.get_world_size()``.
        scale (float): Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all world_size) means a scale of
            10. If None, defaults to ``world_size``.
        patch_optimizer (bool): If True, monkey-patches the ``step`` method of
            the optimizer with the AdaScale ``step`` method.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        patch_optimizer: bool = False,
        num_accumulate_steps: int = 1,
        enabled: bool = True,
    ):
        self._optimizer = optimizer
        self._optimizer_step = optimizer.step
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._world_size: int = world_size if world_size is not None else torch.distributed.get_world_size()
        assert num_accumulate_steps > 0
        self.num_accumulate_steps = num_accumulate_steps
        self._accum_step = 0
        self._enabled = enabled

        if self._world_size * self.num_accumulate_steps <= 1 and enabled:
            raise RuntimeError(
                "AdaScale does not support a single worker with not gradient accumulation."
            )

        self._optimizer.state.setdefault(
            "adascale",
            {
                "grad_sqr_avg": np.ones(len(optimizer.param_groups)),
                "grad_var_avg": np.zeros(len(optimizer.param_groups)),
            },
        )

        for idx, param_group in enumerate(self._optimizer.param_groups):
            for pidx, param in enumerate(param_group["params"]):
                param.register_hook(functools.partial(self._backward_hook, idx))

        self.set_scale(self._world_size if scale is None else scale)
        self._loss_scale = None
        self._fp16_params = None
        self._flat_params = None
        self._final_callback_has_ran = False
        self._initialized = False

        if patch_optimizer:
            self.patch_optimizer()

    def _lazy_init(self):
        if not self._initialized:
            self._initialized = True
            self._fp16_params = []
            self._flat_params = []
            for idx, param_group in enumerate(self._optimizer.param_groups):
                self._fp16_params.append(list(param_group["params"]))
                flat_size = sum(param.numel() for param in self._fp16_params[idx])

                param = self._fp16_params[idx][0]
                flat_params = torch.zeros(
                    (flat_size,), device=param.device, dtype=torch.float32
                )
                flat_params.grad = torch.zeros(
                    (flat_size,), device=param.device, dtype=torch.float32
                )

                param_group["list_params"] = []
                ptr = 0
                for pidx, param in enumerate(param_group["params"]):
                    numel = param.numel()
                    flat_params[ptr : ptr + numel].data.copy_(param.data.view(-1))
                    param_group["list_params"].append(
                        flat_params[ptr : ptr + numel].view_as(param)
                    )
                    param_group["list_params"][pidx].grad = flat_params.grad[
                        ptr : ptr + numel
                    ].view_as(param)

                    ptr += numel

                param_group["params"] = [flat_params]
                self._flat_params.append(flat_params)

    def _update_list_params(self):
        for i, param_group in enumerate(self.param_groups):
            ptr = 0
            flat_params = self._flat_params[i]
            for pidx, param in enumerate(param_group["list_params"]):
                numel = param.numel()
                param_group["list_params"][pidx] = flat_params[
                    ptr : ptr + numel
                ].view_as(param)
                param_group["list_params"][pidx].grad = flat_params.grad[
                    ptr : ptr + numel
                ].view_as(param)

                ptr += numel

    @property
    def theta(self):
        return max(1 - (self._world_size * self.num_accumulate_steps) / 1000, 0)

    @property
    def param_groups(self):
        self._lazy_init()
        return self._optimizer.param_groups

    @property
    def loss_scale(self):
        if self._loss_scale is None:
            self.loss_scale = 1.0
        return self._loss_scale

    @loss_scale.setter
    def loss_scale(self, new_scale: float) -> float:
        self._loss_scale = new_scale

    @property
    def local_scaling_factor(self) -> float:
        return 1.0 / self.loss_scale

    @property
    def state(self) -> Dict[str, np.ndarray]:
        return self._optimizer.state["adascale"]

    @property
    def scale(self) -> float:
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single worker. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        then the scaling factor is 2.5.
        """
        return self._scale

    def set_scale(self, scale: float) -> None:
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Arguments:
            scale (float): New scaling factor to be applied to AdaScale.
        """
        self._scale = scale

    def grad_sqr_avg(self) -> float:
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared in the AdaScale paper).

        Returns (float): Estimate of squared l2-norm.
        """
        return np.sum(self.state["grad_sqr_avg"])

    def grad_var_avg(self) -> float:
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared in the AdaScale paper).

        Returns (float): Estimate of trace of the covariance.
        """
        return np.sum(self.state["grad_var_avg"])

    def gain(self, scale: Optional[float] = None) -> float:
        """
        Current estimate of the AdaScale gain ratio (r_t).

        Arguments:
            scale (float): The batch size scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        scale = self._scale if scale is None else scale
        var = self.grad_var_avg()
        sqr = self.grad_sqr_avg()
        return (var + sqr) / (var / scale + sqr)

    def _update_avg(self, name: str, value: float, factor: float) -> None:
        biased = self.state.get(name + "_biased", 0.0)
        unbias = self.state.get(name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self.state[name + "_biased"] = biased
        self.state[name + "_unbias"] = unbias
        self.state[name] = biased / unbias

    def _compute_grad_sqr_sum(self, grad):
        return _sum_squared_mul(grad, self.local_scaling_factor)

    def _backward_hook(self, idx: int, grad: torch.Tensor) -> None:
        if self._enabled:
            # This method should be invoked once for each parameter during the
            # backward pass, before gradients are synchronized between world_size.
            if self._local_grad_sqr is None:
                self._local_grad_sqr = torch.zeros(
                    (len(self.param_groups),), device=grad.device, dtype=torch.float32,
                )

            self._local_grad_sqr[idx] += self._compute_grad_sqr_sum(
                grad.to(dtype=torch.float32)
            )

        if self._accum_step == (self.num_accumulate_steps - 1):
            self._final_callback_queued = False
            Variable._execution_engine.queue_callback(self._queue_callback)

    def _queue_callback(self) -> None:
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each worker. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _sync_flat_grads(self):
        for pg, model_group in zip(self.param_groups, self._fp16_params):
            for model_param, master_param in zip(model_group, pg["list_params"]):
                if model_param.grad is not None:
                    master_param.grad.data.copy_(model_param.grad.data)

    def _sync_fp16_params(self):
        for pg, model_group in zip(self.param_groups, self._fp16_params):
            for model_param, master_param in zip(model_group, pg["list_params"]):
                model_param.data.copy_(master_param.data)

    def _final_callback(self) -> None:
        # This method should be invoked once for each backward pass, after
        # gradients have been synchronized between each worker.
        self._final_callback_queued = False
        self._final_callback_has_ran = True

        reduce_op = None
        if self._enabled:
            assert isinstance(self._local_grad_sqr, torch.Tensor)
            # self._local_grad_sqr is FP32, sum then div shouldn't overflow.
            reduce_op = torch.distributed.all_reduce(
                self._local_grad_sqr, async_op=True
            )  # SUM

        self._sync_flat_grads()

        if self.num_accumulate_steps != 1:
            for p in self._flat_params:
                p.grad.div_(self.num_accumulate_steps)

        if self._enabled:
            total_grad_sqr = (
                torch.stack(
                    [
                        self._compute_grad_sqr_sum(flat_param.grad)
                        for flat_param in self._flat_params
                    ]
                )
                .cpu()
                .numpy()
            )

            reduce_op.wait()
            local_grad_sqr = self._local_grad_sqr.cpu().numpy()
            self._local_grad_sqr = None

            if np.all(np.isfinite(total_grad_sqr)) and np.all(
                np.isfinite(local_grad_sqr)
            ):
                S = self.scale
                N = self._world_size
                c = self.num_accumulate_steps

                grad_var = S / (c * N - 1) * (local_grad_sqr / (c * N) - total_grad_sqr)

                grad_sqr = total_grad_sqr - grad_var / S

                grad_sqr = np.maximum(grad_sqr, 0.0)
                grad_var = np.maximum(grad_var, 1e-6)
                self._update_avg("grad_sqr_avg", grad_sqr, self.theta)
                self._update_avg("grad_var_avg", grad_var, self.theta)

    def step(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        assert self._accum_step == self.num_accumulate_steps
        assert self._final_callback_has_ran

        self._final_callback_has_ran = False

        if self._enabled:
            initial_lrs = [pg["lr"] for pg in self.param_groups]
            for idx, param_group in enumerate(self.param_groups):
                grad_sqr = float(self.state["grad_sqr_avg"][idx])
                grad_var = float(self.state["grad_var_avg"][idx])
                gain = (grad_var + grad_sqr) / (grad_var / self._scale + grad_sqr)
                param_group["gain"] = gain
                param_group["lr"] *= gain

        res = self._optimizer_step(*args, **kwargs)

        if self._enabled:
            for lr, pg in zip(initial_lrs, self.param_groups):
                pg["lr"] = lr

        self._sync_fp16_params()

        return res

    def inc_accumulate(self):
        self._accum_step += 1
        assert self._accum_step <= self.num_accumulate_steps

    @property
    def accum_step(self):
        return self._accum_step

    def patch_optimizer(self) -> None:
        """
        Monkey-patch the optimizer's step function with :meth:`AdaScale.step`.
        """

        @functools.wraps(self._optimizer.step)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[float]:
            return self.step(*args, **kwargs)

        setattr(self._optimizer, "step", wrapper)

    def zero_grad(self) -> None:
        """Proxy function to optimizer"""
        self._lazy_init()

        for pg in self._fp16_params:
            for param in pg:
                param.grad = None

        for p in self.parameters():
            p.grad.zero_()

        self._accum_step = 0
        self._final_callback_has_ran = False

    def parameters(self):
        return [param for pg in self.param_groups for param in pg["params"]]

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._lazy_init()
        self._optimizer.load_state_dict(state_dict)
        self._update_list_params()
        self._sync_fp16_params()
