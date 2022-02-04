from tqdm import tqdm
import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse