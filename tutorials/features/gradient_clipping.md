# Gradient Clipping

Author: Boxiang Wang

**Prerequisite:**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)

**Example Code**
- [ColossalAI-Examples AMP](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/amp)

**Related Paper**
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)

## Introduction

Gradient clipping is a technique to prevent gradients explosion in deep neural networks. 
By introducing a gradient threshold, gradient clipping makes the gradients norms that exceed the threshold scaling down to match the norm.
Although gradient clipping introduces bias in results, it keep things stable.

### Usage
To use gradient clipping, you can just simply add gradient gradient norm in your configuration file.
```python
clip_grad_norm = 1.0
```

### Hands-On Practice

We provide a [runnable example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_clipping)
to demonstrate gradient clipping. In this example, we set the gradinet clipping vector norm to be 1.0. You can run the script using this command:

```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500  train_with_engine.py
```
