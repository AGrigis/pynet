"""
pynet network helpers overview
==============================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

First checks
------------

In order to test if the 'pynet' package is installed on your machine, you can
check the package version.
"""

import pynet
print(pynet.__version__)

#############################################################################
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly:

import pynet.configure
print(pynet.configure.info())

############################################################################
# Load a network
# --------------
#
# From the available netwoks load the UNet:

import torch
from pynet.models import UNet
from pynet.plotting import plot_net

model = UNet(
    num_classes=2,
    in_channels=1,
    depth=3, 
    start_filts=8,
    up_mode="upsample", 
    merge_mode="concat",
    batchnorm=True)
plot_net(model, shape=(1, 1, 64, 64, 64), static=True, outfileroot=None)


############################################################################
# Inspect a network
# -----------------
#
# Inspect some layers of the UNet:

from pynet.utils import test_model
from pprint import pprint
import numpy as np
from pynet.utils import get_named_layers
from pynet.utils import layer_at
from pynet.plotting import plot_data

out = test_model(model, shape=(1, 1, 64, 64, 64))
layers = get_named_layers(model)
pprint(layers)
hook_x, weight = layer_at(
    model=model,
    layer_name="down.1.doubleconv.conv1-8.16",
    x=torch.FloatTensor(np.random.random((1, 1, 64, 64, 64))))
print(hook_x.shape)
print(weight.shape)
plot_data(hook_x[0, 0], extradata=[im for im in hook_x[0, 1:]])

