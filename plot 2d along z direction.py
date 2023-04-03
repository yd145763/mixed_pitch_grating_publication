# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:14:01 2023

@author: limyu
"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time

file_list = "grating12pitch100", "grating12_11pitch6_4"

file = "grating12_11pitch4_6"
# Load the h5 file
with h5py.File("C:\\Users\\limyu\\Google Drive\\3d plots\\"+file+".h5", 'r') as f:
    # Get the dataset
    dset = f[file]
    # Load the dataset into a numpy array
    arr_3d_loaded = dset[()]

i = 222

x = np.linspace(-20, 80, num=arr_3d_loaded.shape[0])
y = np.linspace(-25, 25, num =arr_3d_loaded.shape[1])
z = np.linspace(-5, 45, num =arr_3d_loaded.shape[2])

z_plane_df = arr_3d_loaded[:,:,i]
df1 = z_plane_df.transpose()
print(z[i])

colorbarmax = df1.max().max()

X,Y = np.meshgrid(x,y)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df1, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
clb.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()

def gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

#define horizontal and vertical cut off lines 
hor_e = z_plane_df[:, int(z_plane_df.shape[1]/2)]
max_ver = np.where(z_plane_df == z_plane_df.max().max())[0][0]
ver_e = z_plane_df[max_ver, :]

#calculate peaks for horizontal line
peaks, _ = find_peaks(hor_e)
peaks_h = x[peaks]
peaks_height_h = hor_e[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_full = peak_widths(hor_e, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]

#gaussian fit for horizontal e-field
A = max(hor_e)
B = 1/(waist_h**2)
max_hor_e_index = int(np.where(hor_e == max(hor_e))[0])
max_hor_e_x = x[max_hor_e_index]
distance_x = x - max_hor_e_x
y_fit = gauss(distance_x, A, B)
#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, hor_e, s = 0.5, color = "black")
ax.plot(distance_x, y_fit, color = "red")

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()


#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(x, hor_e, s = 0.5, color="black")
ax.plot(x[peaks], hor_e[peaks], "o", color="green")
ax.hlines(*results_full_plot[1:], color="blue")

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Peaks", "Beam Waist"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

#calculate peaks for vertical line
peaks, _ = find_peaks(ver_e)
peaks_v = y[peaks]
peaks_height_v = ver_e[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0][0])
results_full = peak_widths(ver_e, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)
waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

#gaussian fit for vertical e-field
A = max(ver_e)
B = 1/(waist_v**2)
max_ver_e_index = int(np.where(ver_e == max(ver_e))[0][0])
max_ver_e_x = y[max_ver_e_index]
distance_y = y - max_ver_e_x
y_fit = gauss(distance_y, A, B)
#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, ver_e, s = 0.5, color = "black")
ax.plot(distance_y, y_fit, color = "red")

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(y, ver_e, s = 0.5, color="black")
ax.plot(y[peaks], ver_e[peaks], "o", color="green")
ax.hlines(*results_full_plot[1:], color="blue")

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Peaks", "Beam Waist"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()
    