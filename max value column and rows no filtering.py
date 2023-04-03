# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:28:52 2023

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:06:44 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
import statistics


df = pd.read_csv(r"C:\Users\limyu\Google Drive\different mixture\different mixture 4 beams further from ref and a beam 3800-3900cnts 4x 90mA.csv",header=None, sep=",")
    
df=df.dropna(axis=1)
df_r = df.iloc[14:38, 214:248]
df_r = df_r.reset_index(drop=True)
df_r.columns = range(df_r.shape[1])
H = np.arange(0,24,1)
V = np.arange(0,34,1)

max_df_r = df_r.max().max()
is_duplicate = df.eq(max_df_r)

row_idxs, col_idxs = np.where(df_r == max_df_r)
print(len(row_idxs))
print(len(col_idxs))
for row_idx, col_idx in zip(row_idxs, col_idxs):
    print(row_idx)
    print(col_idx)
    print(" ")
xr = np.linspace(0, 990, num=34)
xr = xr/20
yr = np.linspace(0, 690, num=24)
yr = yr/20
colorbarmax = 5000
colorbartick = 5

Xr,Yr = np.meshgrid(xr,yr)
df_r = df_r.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr,Yr,df_r, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 200)).tolist())
clb.ax.set_title('Photon/s', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.axhline(y=yr[int(row_idx)], color='r', linestyle = "--")
ax.axvline(x=xr[int(col_idx)], color='g', linestyle = "--")
plt.show()
plt.close()


H = np.arange(0,24,1)
V = np.arange(0,34,1)

V1,H1 = np.meshgrid(V,H)
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(V1,H1,df_r, 200, zdir='z', offset=-100, cmap='hot')
ax.axhline(y=int(row_idx), color='r')
ax.axvline(x=int(col_idx), color='g')
plt.show()
plt.close()