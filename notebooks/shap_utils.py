import librosa
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
def plot_shap_spec(shap_values, spec, sr, pred, true, shap_idx=None):
  """
  Function to plot the SHAP values for one spectrogram and one prediction label.
  :param shap_values: array of shap values to plot
  :param spec: original spectrogram of audio file
  :param sr: sampling rate
  :param pred: prediction label
  :param true: ground truth label
  :param shap_idx: index of the prediction index to display shap values for.
         Leave as None to plot the shap values for the top prediction.
  """

  figsize = [9, 2.5]

  pred_idx = LABELS.index(pred)

  # Define colors for the SHAP-like colormap
  colors = []
  for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,j))
  for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,j))

  # Create the colormap
  shap_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

  # Setup figure
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

  #plot the spectrogram image
  spec = np.squeeze(spec)
  librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', cmap = 'viridis', ax=axes[0])
  axes[0].set_title(f'Label: {true} | Pred: {pred}')

  # plot the shap values
  shap_idx = shap_idx if shap_idx is not None else pred_idx
  sv = np.mean(shap_values[:, :, :, pred_idx], axis=-1) # aggregate the channels
  lim = max(abs(sv.min()), abs(sv.max()))
  librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', cmap = 'Greys_r', ax=axes[1], alpha=0.20)
  im = librosa.display.specshow(sv*2, sr=sr, x_axis='time', y_axis='mel', cmap = shap_cmap, ax=axes[1], vmin=-lim, vmax=lim)
  axes[1].set_title(f'Shap Values: {LABELS[shap_idx]}')
  axes[1].set_yticks([])
  axes[1].set_ylabel('')

  # setup colorbar
  width = 0.725
  left = (1 - width) / 2 + .01
  cax = fig.add_axes([left, -0.1, width, 0.03])  # [left, bottom, width, height]
  cb = fig.colorbar(im, cax=cax, label="Importance", orientation="horizontal",
                    aspect=figsize[0] / 0.2)
  cb.outline.set_visible(False)