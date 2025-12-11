class Conv3d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.Conv3d,
    input: Tensor) -> Tensor:
    weight = self.weight
    input0 = torch._convolution(input, weight, None, [1, 2, 2], [2, 3, 3], [1, 1, 1], False, [0, 0, 0], 1, False, False, True, True)
    return input0
class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    input: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    input1 = torch._convolution(input, weight, bias, [2, 2], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input1
