class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.Conv3d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.BatchNorm3d
  __annotations__["2"] = __torch__.torch.nn.modules.activation.ReLU
  __annotations__["3"] = __torch__.torch.nn.modules.pooling.MaxPool3d
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    input: Tensor) -> Tensor:
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _4 = (_1).forward((_0).forward(input, ), )
    return (_3).forward((_2).forward(_4, ), )
