import os
#import cv2
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import average_precision_score as aps
from sklearn.preprocessing import MinMaxScaler

from scipy import interp

def py_map2jpg(imgmap, rang, colorMap):
  if rang is None:
    rang = [np.min(imgmap), np.max(imgmap)]
  heatmap_x = np.round(imgmap*255).astype(np.uint8)
  return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)
#enddef

def im2double(im):
  #return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
  return MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(im) 


#enddef

def draw(image, heatmap, title, saved_path=None):
  h,w,c = image.shape
  resize_heatmap = cv2.resize(heatmap, (w,h))
  curHeatMap = cv2.resize(im2double(resize_heatmap),(w,h))
  curHeatMap = im2double(curHeatMap)
  curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
  image = cv2.resize(image, (w, h))
  curHeatMap = im2double(image)*0.4+im2double(curHeatMap)*0.6
  if saved_path is not None:
    cv2.imwrite(saved_path, curHeatMap*255)
    cv2.imshow('HeatMap:%s'%title, curHeatMap)
    cv2.waitKey(0)
#enddef

def calc_cam_map(activations, weights, choosed_labels):
  weights = np.squeeze(weights)
  activations = np.squeeze(activations)
  camp_maps = []
  for choosed in choosed_labels:
    coeff = weights.T[choosed].flatten()
    cam_map = np.sum(activations * coeff, axis=-1)
    camp_maps.append(cam_map)
  return camp_maps

def plot_cam_map(image, activations, weights, choosed_labels, saved_paths=None):
  if saved_paths is not None:
    assert len(choosed_labels)==len(saved_paths), 'number of labels must be samed with paths'
  #cvimage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
  cam_maps = calc_cam_map(activations, weights, choosed_labels)
  b,h,t,c = image.shape
  for k, cam_map in enumerate(cam_maps):
    im = Image.fromarray(np.array(cam_maps * 255, np.uint8))
    im = im.resize(()) 
    if saved_paths is not None:  
      pass
      #draw(cvimage, cam_map, "label:%d"%choosed_labels[k], saved_path=saved_paths[k])
    else:
      pass
      #draw(cvimage, cam_map, "label:%d"%choosed_labels[k])
  #draw(cv2.imread(current_path), hmap, "%s-%s"%(current_path,name))
   

"""
def plot_confusion_matrix(confusion_mat, labels, title, cmap=plt.cm.binary):
  plt.imshow(confusion_mat, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, fontsize=12)
  plt.yticks(xlocations, labels, fontsize=12, rotation=90)
  plt.ylabel('True Label', fontsize=12)
  plt.xlabel('Predicted Label', fontsize=12)
  #plt.savefig('../Data/confusion_matrix.png', format='png')
  plt.show()

def plot_normal_confusion_matrix(true_label, pred_label, labels, title):
  tick_marks = np.array(range(len(labels))) + 0.5
  cm = confusion_matrix(true_label, pred_label)
  np.set_printoptions(precision=2)
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  #print cm_normalized
  plt.figure(figsize=(6, 4), dpi=120)

  ind_array = np.arange(len(labels))
  x, y = np.meshgrid(ind_array, ind_array)

  for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
      plt.text(x_val, y_val, "%0.2f" % (c,), 
      color='red', fontsize=12, va='center', ha='center')
  # offset the tick
  plt.gca().set_xticks(tick_marks, minor=True)
  plt.gca().set_yticks(tick_marks, minor=True)
  plt.gca().xaxis.set_ticks_position('none')
  plt.gca().yaxis.set_ticks_position('none')
  plt.grid(True, which='minor', linestyle='-')
  plt.gcf().subplots_adjust(bottom=0.15)
  plot_confusion_matrix(cm_normalized, labels, '%s-Normalized confusion matrix'%title)
#enddef

def plot_roc(true_label, pred_probs, curve_descs, title):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100) # x: mean_fpr.
  for k in range(len(pred_probs)):
    fpr, tpr, thresholds = roc_curve(true_label[k,:], pred_probs[k, :])
    tprs.append(interp(mean_fpr, fpr, tpr)) # y: interpolated tprs.
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC %s (AUC=%0.2F)'%(curve_descs[k], roc_auc))  

  plt.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, color='b', 
    label = r'Mean ROC (AUC=%0.2f $\pm$ %0.2f)'%(mean_auc, std_auc), lw=2, alpha=.8)  

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
  plt.fill_between(mean_fpr, tprs_lower, tprs_upper, 
    color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('%s Receiver Operating Characteristic Curves'%title)
  plt.legend(loc='lower right')
  plt.show()
#enddef


def plot_prc(true_label, pred_probs, curve_descs, title):
  for k in range(len(pred_probs)):
    precision, recall, thresholds = pr_curve(true_label[k,:], pred_probs[k,:])
    average_precision_score = aps(true_label[k], pred_probs[k,:])
    plt.plot(recall, precision, lw=1, alpha=0.8, 
      label='PRC %s (APS=%0.2F)'%(curve_descs[k], average_precision_score))  

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('%s Precision Recall Curves'%title)
  plt.legend(loc='lower right')
  plt.show()
#enddef
"""
