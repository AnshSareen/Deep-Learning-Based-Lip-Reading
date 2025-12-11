class LipReadingModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  frontend : __torch__.torch.nn.modules.container.Sequential
  backend : __torch__.torch.nn.modules.container.___torch_mangle_4.Sequential
  fc : __torch__.torch.nn.modules.linear.Linear
  dropout : __torch__.torch.nn.modules.dropout.Dropout
  lstm : __torch__.torch.nn.modules.rnn.LSTM
  layer_norm : __torch__.torch.nn.modules.normalization.LayerNorm
  classifier : __torch__.torch.nn.modules.linear.___torch_mangle_5.Linear
  def forward(self: __torch__.LipReadingModel,
    x: Tensor) -> Tensor:
    classifier = self.classifier
    layer_norm = self.layer_norm
    lstm = self.lstm
    dropout = self.dropout
    fc = self.fc
    backend = self.backend
    frontend = self.frontend
    input = torch.permute(x, [0, 2, 1, 3, 4])
    _0 = (frontend).forward(input, )
    B = ops.prim.NumToTensor(torch.size(_0, 0))
    _1 = int(B)
    C = ops.prim.NumToTensor(torch.size(_0, 1))
    _2 = int(C)
    T = ops.prim.NumToTensor(torch.size(_0, 2))
    _3 = int(T)
    H = ops.prim.NumToTensor(torch.size(_0, 3))
    _4 = int(H)
    W = ops.prim.NumToTensor(torch.size(_0, 4))
    _5 = int(W)
    _6 = torch.contiguous(torch.permute(_0, [0, 2, 1, 3, 4]))
    input0 = torch.view(_6, [int(torch.mul(B, T)), _2, _4, _5])
    input1 = torch.view((backend).forward(input0, ), [_1, _3, -1])
    _7 = (dropout).forward((fc).forward(input1, ), )
    _8 = (layer_norm).forward((lstm).forward(_7, ), )
    _9 = torch.permute((classifier).forward(_8, ), [1, 0, 2])
    return _9
