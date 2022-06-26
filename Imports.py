## Imports

try:
# %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
!sudo apt-get install graphviz
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow.keras as tk

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

!pip install tensorflow-addons
import tensorflow_addons as tfa

tk = tf.keras;
tkl = tk.layers;



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.ticker as mticker

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.manifold import TSNE
import os, re, glob2, pickle

#from keras.engine import  Model
from keras.layers import Input
#from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
#from keras_vggface import utils

import matplotlib.pyplot as plt
%pylab inline

from google.colab import drive
drive.mount('/content/drive')
