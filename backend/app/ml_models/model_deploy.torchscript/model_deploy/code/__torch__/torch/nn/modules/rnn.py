class LSTM(Module):
  __parameters__ = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse", "weight_ih_l1", "weight_hh_l1", "bias_ih_l1", "bias_hh_l1", "weight_ih_l1_reverse", "weight_hh_l1_reverse", "bias_ih_l1_reverse", "bias_hh_l1_reverse", ]
  __buffers__ = []
  weight_ih_l0 : Tensor
  weight_hh_l0 : Tensor
  bias_ih_l0 : Tensor
  bias_hh_l0 : Tensor
  weight_ih_l0_reverse : Tensor
  weight_hh_l0_reverse : Tensor
  bias_ih_l0_reverse : Tensor
  bias_hh_l0_reverse : Tensor
  weight_ih_l1 : Tensor
  weight_hh_l1 : Tensor
  bias_ih_l1 : Tensor
  bias_hh_l1 : Tensor
  weight_ih_l1_reverse : Tensor
  weight_hh_l1_reverse : Tensor
  bias_ih_l1_reverse : Tensor
  bias_hh_l1_reverse : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.rnn.LSTM,
    argument_1: Tensor) -> Tensor:
    bias_hh_l1_reverse = self.bias_hh_l1_reverse
    bias_ih_l1_reverse = self.bias_ih_l1_reverse
    weight_hh_l1_reverse = self.weight_hh_l1_reverse
    weight_ih_l1_reverse = self.weight_ih_l1_reverse
    bias_hh_l1 = self.bias_hh_l1
    bias_ih_l1 = self.bias_ih_l1
    weight_hh_l1 = self.weight_hh_l1
    weight_ih_l1 = self.weight_ih_l1
    bias_hh_l0_reverse = self.bias_hh_l0_reverse
    bias_ih_l0_reverse = self.bias_ih_l0_reverse
    weight_hh_l0_reverse = self.weight_hh_l0_reverse
    weight_ih_l0_reverse = self.weight_ih_l0_reverse
    bias_hh_l0 = self.bias_hh_l0
    bias_ih_l0 = self.bias_ih_l0
    weight_hh_l0 = self.weight_hh_l0
    weight_ih_l0 = self.weight_ih_l0
    max_batch_size = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _0 = int(max_batch_size)
    hx = torch.zeros([4, int(max_batch_size), 512], dtype=6, layout=None, device=torch.device("cuda:0"), pin_memory=False)
    hx0 = torch.zeros([4, _0, 512], dtype=6, layout=None, device=torch.device("cuda:0"), pin_memory=False)
    _1 = [hx, hx0]
    _2 = [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse, weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1, weight_ih_l1_reverse, weight_hh_l1_reverse, bias_ih_l1_reverse, bias_hh_l1_reverse]
    input, _3, _4 = torch.lstm(argument_1, _1, _2, True, 2, 0.29999999999999999, False, True, True)
    return input
