import os
import tensorflow as tf

import model

# Converts the raw audio data into a float32 tensor. Normalizes it to the [-1.0, 1.0] range. 
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # This removes the extra axis for stereo channels to have a truly 1 dimensional wavefor data. 
    return tf.squeeze(audio, axis=-1)

# Gets the label for the particular file (i.e Go, No, Stop, etc...)
def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  return parts[-2]

# This creates a tuple of (waveform data, data label)
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label