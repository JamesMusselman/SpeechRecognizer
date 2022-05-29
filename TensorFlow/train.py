import os
import tensorflow as tf
import numpy as np
import model

from spectogram import *
from audio import *

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=model._autotune)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=model._autotune)
  return output_ds