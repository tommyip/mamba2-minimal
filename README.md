# mamba2-minimal

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

![Mamba-2](https://github.com/state-spaces/mamba/blob/f9dbb4fdb2705d71282e0db184d177c6375623f0/assets/ssd_algorithm.png)
> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> Paper: https://arxiv.org/abs/2405.21060

This implementation is device agnostic and have been tested to work on the CPU and MPS (Metal Performance Shaders) backends. The model's output logits follow the same distribution as the reference implementation but are not equal at the bit level. 

## Usage

Install dependencies (`torch`, `einops` and `transformers`):

```
pip install -r requirements.txt
```

See [demo.ipynb](./demo.ipynb) for using Mamba-2 as part of an end-to-end language model with pretrained weights for text generation.

The core Mamba-2 model can be used as follows:

```py
import torch

from mamba2 import Mamba2, Mamba2Config

config = Mamba2Config(d_model=768)
model = Mamba2(config)

x = torch.randn(2, 64, 768)  # (batch, seqlen, d_model)
y = model(x)  # same shape as x
```


## TODOs

- [x] Constant time (wrt sequence length) autoregressive inference
- [ ] Remove dependency on `einops` (depends on whether resulting code is still readable)

## Credits

* [Albert Gu], [Tri Dao] - authors of the Mamba-2 architecture
  * [paper]
  * [reference implementation]
  * [blog post]
* [John Ma] - author of [johnma2006/mamba-minimal], who inspired this repo


[Albert Gu]: https://github.com/albertfgu
[Tri Dao]: https://github.com/tridao
[paper]: https://arxiv.org/abs/2405.21060
[reference implementation]: https://github.com/state-spaces/mamba
[blog post]: https://tridao.me/blog/2024/mamba2-part1-model/
[John Ma]: https://github.com/johnma2006
[johnma2006/mamba-minimal]: https://github.com/johnma2006/mamba-minimal
