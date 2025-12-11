class MaxPool3d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.MaxPool3d,
    argument_1: Tensor) -> Tensor:
    x = torch.max_pool3d(argument_1, [1, 3, 3], [1, 2, 2], [0, 1, 1], [1, 1, 1])
    return x
class AdaptiveAvgPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d,
    argument_1: Tensor) -> Tensor:
    x = torch.adaptive_avg_pool2d(argument_1, [1, 1])
    return x
