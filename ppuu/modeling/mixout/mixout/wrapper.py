import torch
from torch import nn


class MixoutWrapper(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, module: nn.Module):
        """
        Implementation of Mixout (https://arxiv.org/abs/1909.11299).
        Use with:
        >>> mixout_model = model.apply(MixoutWrapper).
        """
        # duplicate all the parameters, making copies of them and freezing them
        module._names = []
        module._params_orig = dict()
        _params_learned = nn.ParameterDict()
        for n, q in list(module.named_parameters(recurse=False)):
            c = q.clone().detach()
            c.requires_grad = False
            module._params_orig[n] = c
            _params_learned[n] = q
            module._names.append(n)
            delattr(module, n)
            setattr(module, n, c)
        if module._names:
            module._params_learned = _params_learned

        self.hook = Hook(self.p)

        module.register_forward_pre_hook(self.hook)
        return module


class Hook(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, module, input):
        for n in module._names:
            v = torch.nn.Parameter(self.mixout(module, n))
            setattr(module, n, v)

    def mixout(self, module, n):
        if module.training:
            o = module._params_orig[n]
            mask = (torch.rand_like(o) < self.p).type_as(o)
            return (
                mask * module._params_orig[n]
                + (1 - mask) * module._params_learned[n]
                - self.p * module._params_orig[n]
            ) / (1 - self.p)
        else:
            return module._params_learned[n]
