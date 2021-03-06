# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
A module with common functions.
"""


# System import
import os 

# Third party imports
import torch
import numpy as np


ALLOWED_LAYERS = [
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.Linear
]


def logo():
    """ pySAP logo is ascii art using Big Money-ne.

    Returns
    -------
    logo: str
        the logo.
    """
    logo = r"""
                                           /$$    
                                          | $$    
  /$$$$$$  /$$   /$$ /$$$$$$$   /$$$$$$  /$$$$$$  
 /$$__  $$| $$  | $$| $$__  $$ /$$__  $$|_  $$_/  
| $$  \ $$| $$  | $$| $$  \ $$| $$$$$$$$  | $$    
| $$  | $$| $$  | $$| $$  | $$| $$_____/  | $$ /$$
| $$$$$$$/|  $$$$$$$| $$  | $$|  $$$$$$$  |  $$$$/
| $$____/  \____  $$|__/  |__/ \_______/   \___/  
| $$       /$$  | $$                              
| $$      |  $$$$$$/                              
|__/       \______/ """
    return logo


def test_model(model, shape):
    """ Simple function to test a model.

    Parameters
    ----------
    model: Net
        the network model.
    shape: list of int
        the shape of a classical input batch dataset.
    """
    x = torch.autograd.Variable(torch.FloatTensor(np.random.random(shape)))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    return out


def checkpoint(model, epoch, fold, outdir):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: Net
        the network model.
    epoch: int
        the epoch index.
    fold: int
        the fold index.
    outdir: str
        the destination directory where a 'model_<fold>_epoch_<epoch>.pth'
        file will be generated.
    """
    outfile = os.path.join(
        outdir, "model_{0}_epoch_{1}.pth".format(fold, epoch))
    torch.save(model, outfile)
    return outfile


def get_named_layers(model, allowed_layers=ALLOWED_LAYERS):
    """ Function that returned a dictionary with named layers.

    Parameters
    ----------
    model: Net
        the network model.
    allowed_layers: list of str, default ALLOWED_LAYERS
        the allowed modules.

    Returns
    -------
    layers: dict
        the named layers.
    """
    layers = {}
    for name, mod in model.named_modules():
        name = name.replace("ops.", "")
        for klass in allowed_layers:        
            if isinstance(mod, klass):
                if hasattr(mod, "in_channels") and hasattr(mod, "out_channels"):
                    name = "{0}-{1}.{2}".format(
                        name, mod.in_channels, mod.out_channels)
                elif hasattr(mod, "num_features"):
                    name = "{0}-{1}".format(name, mod.num_features)
                else:
                    raise ValueError("Layer of type '{0}' is not yet "
                                     "supported.".format(klass.__name__))
                layers[name] = mod
    return layers


def layer_at(model, layer_name, x, allowed_layers=ALLOWED_LAYERS):
    """ Access intermediate layers of pretrined network.

    Parameters
    ----------
    model: Net
        the network model.
    layer_name: str
        the layer name to be inspected.
    x: torch.Tensor
        an input tensor.
    allowed_layers: list of str, default ALLOWED_LAYERS
        the allowed modules.

    Returns
    -------
    hook_x: torch.Tensor
        the tensor generated at the requested location.
    weight: torch.Tensor
        the layer associated weight.
    """
    layers = get_named_layers(model)
    layer = layers[layer_name]
    global hook_x
    def hook(module, inp, out):
        """ Define hook.
        """
        print(
            "layer:", type(module),
            "\ninput:", type(inp),
                "\n   len:", len(inp),
                "\n   type:", type(inp[0]),
                "\n   data size:", inp[0].data.size(),
                "\n   data type:", inp[0].data.type(),
            "\noutput:", type(out),
                "\n   data size:", out.data.size(),
                "\n   data type:", out.data.type())
        global hook_x
        hook_x = out.data
    _hook = layer.register_forward_hook(hook)
    _ = model(x)
    _hook.remove()
    return hook_x.numpy(), layer.weight.detach().numpy()


