import os
import tensorflow as tf
import numpy as np
import model

from spectogram import *
from audio import *

def get_spectrogram(waveform):
  # Zero-padding for any audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)

  waveform = tf.cast(waveform, dtype=tf.float32)
  waveform_padded = tf.concat([waveform, zero_padding], 0)

# Applies short-time Fourier transform to convert 
  spectrogram = tf.signal.stft(
      waveform_padded, frame_length=255, frame_step=128)
      
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
    
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.math.argmax(label == model._commands)
  return spectrogram, label_id

  