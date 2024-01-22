import torch 


class AddGaussianNoise:

  def __init__(self, mean=0., std=1.):
      self.std = std
      self.mean = mean

  def __call__(self, tensor):
      return tensor + torch.randn(tensor.size()) * self.std + self.mean

  def __repr__(self):
      return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class Clipper:

  def __init__(self, low=0., high=1.):
    self.low = low
    self.high = high

  def __call__(self, tensor):
    return torch.clip(tensor, self.low, self.high)

  def __repr__(self):
    return self.__class__.__name__ + f"(low={self.low}, high={self.high})"