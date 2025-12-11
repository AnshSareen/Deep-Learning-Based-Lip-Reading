class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_0.ReLU
  __annotations__["3"] = __torch__.torch.nn.modules.conv.___torch_mangle_1.Conv2d
  __annotations__["4"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_2.BatchNorm2d
  __annotations__["5"] = __torch__.torch.nn.modules.activation.___torch_mangle_3.ReLU
  __annotations__["6"] = __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_4.Sequential,
    input: Tensor) -> Tensor:
    _6 = getattr(self, "6")
    _5 = getattr(self, "5")
    _4 = getattr(self, "4")
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _7 = (_1).forward((_0).forward(input, ), )
    _8 = (_4).forward((_3).forward((_2).forward(_7, ), ), )
    return (_6).forward((_5).forward(_8, ), )
