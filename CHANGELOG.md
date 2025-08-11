# Change Log

* [2025-08-11] Added support for [Adaptive Computation Pruning](https://openreview.net/forum?id=xNj14CY5S1). We also simplied the dependencies when using this repo as a library.
* [2025-03-31] In our paper experiments we accidentally share the RMSNorm scaling parameters across heads in the QK-norm implementation. Normally there should be a total of `num_heads * head_dim` scaling parameters but in our experiments there were only `head_dim` scaling parameters shared across the `num_heads` heads. We have verified that this has no observable impact on performance. Nevertheless, we have added a `qk_norm_share_param_across_head` argument that controls this behavior and set the default value to `False` because it makes more sense. Note that for backward compatibility, in our provided checkpoints `qk_norm_share_param_across_head` is still set to `True` (otherwise the weights cannot be loaded).

