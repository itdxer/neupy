from .base import *
from .graph import *
from .activations import *
from .convolutions import *
from .pooling import *
from .stochastic import *
from .normalization import *
from .merge import *
from .reshape import *
from .embedding import *
from .recurrent import *


# Extra aliases for the layers
MaxPool = MaxPooling
AvgPool = AveragePooling
GlobalPool = GlobalPooling

Conv = Convolution
BN = BatchNorm
GN = GroupNorm
