import warnings

import torch


class BlowupSchedule(torch.optim.lr_scheduler._LRScheduler):
    """This only really works if you call it per step, not per epoch"""

    def __init__(
        self,
        optimizer,
        warmup_rate,
        save_interval,
        loss_threshold=10,
        last_epoch=-1,
    ):
        # The warmup rate is the number of updates it takes
        # to order of magnitude the learning rate. Higher is slower
        self.warmup_rate = 10 ** (1 / warmup_rate)
        self.loss_threshold = loss_threshold
        self.save_interval = save_interval
        self.warmup_updates = float("inf")
        self.warmup_end_mod = 0
        self.just_now = None
        self.last_good = None
        super().__init__(optimizer, last_epoch)
        self.just_now = self._checkpoint()

    def _lrmod(self, f):
        return [f(group["lr"]) for group in self.optimizer.param_groups]

    def _blown_up(self):
        return self.last_loss > self.loss_threshold
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if torch.isnan(p).any().item():
                    return True
        return False

    def _checkpoint(self):
        return (
            self.optimizer.state_dict(),
            [
                [p.cpu() for p in group["params"]]
                for group in self.optimizer.param_groups
            ],
        )

    def _reload(self, ckpt):
        optim_state_dict, params = ckpt
        self.optimizer.load_state_dict(optim_state_dict)
        for group, savegroup in zip(self.optimizer.param_groups, params):
            for p, savep in zip(group["params"], savegroup):
                p.detach().copy_(savep)

    def step(self, loss=0):
        self.last_loss = loss
        super().step()

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        step = self.last_epoch

        if self.last_good is not None and self._blown_up():
            self.warmup_updates = step + (step % self.save_interval)
            # We hit instablity!
            # Do some stuff like reload the optimizer
            self._reload(self.last_good)
            self.just_now = None
            self.last_good = None
        if self.just_now is not None and (step + 1) % self.save_interval == 0:
            self.last_good = self.just_now
            self.just_now = self._checkpoint()

        # While you're warming up, do linear increase:
        if step < self.warmup_updates:
            return self._lrmod(lambda lr: lr * self.warmup_rate)
        elif step == self.warmup_updates:
            self.warmup_end_mod = self._lrmod(
                lambda lr: lr * self.warmup_updates ** 0.5
            )
        # After you're done, do inverse sqrt decay
        return [lr * step ** -0.5 for lr in self.warmup_end_mod]
