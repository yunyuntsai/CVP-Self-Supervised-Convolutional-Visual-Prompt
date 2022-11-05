import logging

from copy import deepcopy

import torch
import torch.nn as nn
logger = logging.getLogger(__name__)


class Norm(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_model(model, eps, momentum, reset_stats,
                                     no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        outputs, _ = self.model(x)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def collect_stats(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names


def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_optimizer(params):
    # ------------------------------- Optimizer options ------------------------- #
    # Number of updates per batch
    STEPS = 1
    # Learning rate
    LR = 1e-3
    # Choices: Adam, SGD
    METHOD = 'Adam'
    # Beta
    BETA = 0.9
    # Momentum
    MOMENTUM = 0.9
    # Momentum dampening
    DAMPENING = 0.0
    # Nesterov momentum
    NESTEROV = True
    # L2 regularization
    WD = 0.0

    if METHOD == 'Adam':
        return optim.Adam(params,
                    lr=LR,
                    betas=(BETA, 0.999),
                    weight_decay=WD)
    elif METHOD == 'SGD':
        return optim.SGD(params,
                   lr=LR,
                   momentum=MOMENTUM,
                   dampening=DAMPENING,
                   weight_decay=WD,
                   nesterov=NESTEROV)
    else:
        raise NotImplementedError