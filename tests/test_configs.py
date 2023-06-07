import itertools
import os
import pathlib
import tempfile

import gin
import pytest
import torch

from msprior.scripted import ScriptedPrior

torch.set_grad_enabled(False)

configs = map(str, pathlib.Path("msprior/configs").glob("*.gin"))
configs = filter(lambda x:"flattened" not in x, configs)
configs = filter(lambda x:"rwkv" not in x, configs)
configs = list(configs)

names = map(os.path.basename, configs)
names = map(lambda x: os.path.splitext(x)[0], names)

names = itertools.product(names, ["continuous", "discrete"],
                          ["listen", "generate"])
names = map(lambda x: " ".join(x), names)

continuous = [True, False]
listen = [True, False]
configs = itertools.product(configs, continuous, listen)


@pytest.mark.parametrize("config,continuous,listen", configs, ids=names)
def test_config(config, continuous, listen):
    gin.clear_config()
    gin.parse_config_file(config)

    model = ScriptedPrior(
        from_continuous=continuous,
        initial_listen=listen,
    )

    if listen and not continuous:
        x = torch.randint(0, 16, (1, model.forward_params[0], 8)).float()
        y = model(x)
        assert torch.allclose(y[:, :-1].float(), x[:, :y.shape[1] - 1].float())

        x = torch.randint(0, 16, (4, model.forward_params[0], 8)).float()
        y = model(x)
        assert torch.allclose(y[:, :-1].float(), x[:, :y.shape[1] - 1].float())

    with tempfile.TemporaryDirectory() as tmp:
        model.export_to_ts(os.path.join(tmp, "model.ts"))
