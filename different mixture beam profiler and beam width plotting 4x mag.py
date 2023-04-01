# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:21:33 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths

master_data_verticle = pd.DataFrame([])
master_data_horizontal = pd.DataFrame([])
full_width_horizontal = []
full_width_verticle = []
half_width_horizontal = []
half_width_verticle = []
verticle_peaks = []
horizontal_peaks = []
horizontal_peaks_position = []
verticle_peaks_position = []
horizontal_width_cut = []
verticle_width_cut = []
cut = 1200

df = pd.read_csv(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\2nd march different radius lower 15 20 30 um aligning the ref 3800-3900cnts 4x 55mA mA_0001.ascii.csv",header=None, sep=",")
    
df=df.dropna(axis=1)
print(df)

verticle_peaks = []
verticle_half = []
verticle_full = []

horizontal_peaks = []
horizontal_half = []
horizontal_full = []

x = np.linspace(0, 9570, num=320)
x = x/20
y = np.linspace(0, 7650, num=256)
y = y/20
colorbarmax = 4500
colorbartick = 9

X,Y = np.meshgrid(x,y)
df1 = df.to_numpy()

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\radius1_lower_everything_feb2023_feb2023.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, df1, cmap='hot')
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=13)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=15)
ax.set_zlabel('Photon/s', fontsize=18, fontweight="bold", labelpad=15)


ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.zaxis.label.set_fontsize(18)
ax.zaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_zticklabels(ax.get_zticks(), weight='bold')
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\radius1_lower_everything_3D_feb2023.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

df_r1 = df.iloc[222:246, 206:240]
df_r1 = df_r1.reset_index(drop=True)
df_r1.columns = range(df_r1.shape[1])
df_r1_hor = df_r1.iloc[12,:]
df_r1_ver = df_r1.iloc[:, 18]
xr1 = np.linspace(0, 990, num=34)
xr1 = xr1/20
yr1 = np.linspace(0, 690, num=24)
yr1 = yr1/20
colorbarmax = 5000
colorbartick = 9

print(df_r1.max().max())
row = df_r1.isin([3020]).any(axis=1).idxmax()
col = df_r1.isin([3020]).any(axis=0).idxmax()
print(row)
print(col)

Xr1,Yr1 = np.meshgrid(xr1,yr1)
df_r1 = df_r1.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr1,Yr1,df_r1, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
ax.axhline(y=yr1[12], color='r', linestyle = "--")
ax.axvline(x=xr1[18], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\15µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr1, df_r1_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr1), max(xr1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r1 = y_fit 
peaks, _ = find_peaks(h_r1)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r1, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r1, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)

horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

print("15µmhorizontalr1")
print(max(results_half_plot[0]))
print(" ")

print("15µmhorizontalfullr1")
print(max(results_full_plot[0]))
print(" ")

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r1 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r1)
ax2.plot(peaks_h, h_r1[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\15µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

ax2 = plt.axes()
ax2.plot(x_new, h_r1)
ax2.plot(peaks_h, h_r1[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width"], prop={'weight': 'bold'})
plt.show()
plt.close()



full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r1))
horizontal_peaks_position.append(x_new[np.argmax(h_r1)])
master_data_horizontal["15um"] = h_r1

#plot verticle
ax2 = plt.axes()
tck = interpolate.splrep(yr1, df_r1_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr1), max(yr1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r1 = y_fit 
peaks, _ = find_peaks(v_r1)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r1, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r1, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("15µmverticler1")
print(max(results_half_plot[0]))
print(" ")

print("15µmverticlefullr1")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r1 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r1)
ax2.plot(peaks_v, v_r1[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\15µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

ax2 = plt.axes()
ax2.plot(x_new, v_r1)
ax2.plot(peaks_v, v_r1[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width"], prop={'weight': 'bold'})
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r1))
verticle_peaks_position.append(x_new[np.argmax(v_r1)])
master_data_verticle["15um"] = v_r1

df_r2 = df.iloc[82:106, 206:240]
df_r2_hor = df_r2.iloc[12,:]
df_r2_ver = df_r2.iloc[:, 16]
xr2 = np.linspace(0, 990, num=34)
xr2 = xr2/20
yr2 = np.linspace(0, 690, num=24)
yr2 = yr2/20
colorbarmax = 5000
colorbartick = 5

Xr2,Yr2 = np.meshgrid(xr2,yr2)
df_r2 = df_r2.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr2,Yr2,df_r2, 200, zdir='z', offset=-100, cmap='hot')
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
ax.axhline(y=yr2[12], color='r', linestyle = "--")
ax.axvline(x=xr2[16], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\20µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr2, df_r2_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr2), max(xr2), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r2 = y_fit 
peaks, _ = find_peaks(h_r2)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r2, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r2, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("20µmhorizontalr2")
print(max(results_half_plot[0]))
print(" ")

print("20µmhorizontalfullr2")
print(max(results_full_plot[0]))
print(" ")
horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r2 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r2)
ax2.plot(peaks_h, h_r2[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")


ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\20µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r2))
horizontal_peaks_position.append(x_new[np.argmax(h_r2)])
master_data_horizontal["20um"] = h_r2

#plot verticle
ax2 = plt.axes()
tck = interpolate.splrep(yr2, df_r2_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr2), max(yr2), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r2 = y_fit 
peaks, _ = find_peaks(v_r2)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r2, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r2, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("20µmverticler2")
print(max(results_half_plot[0]))
print(" ")

print("20µmverticlefullr2")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r2 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r2)
ax2.plot(peaks_v, v_r2[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")


ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\20µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r2))
verticle_peaks_position.append(x_new[np.argmax(v_r2)])
master_data_verticle["20um"] = v_r2

df_r3 = df.iloc[14:38, 206:240]
df_r3_hor = df_r3.iloc[11,:]
df_r3_ver = df_r3.iloc[:, 16]
xr3 = np.linspace(0, 990, num=34)
xr3 = xr3/20
yr3 = np.linspace(0, 690, num=24)
yr3 = yr3/20
colorbarmax = 5000
colorbartick = 5

Xr3,Yr3 = np.meshgrid(xr3,yr3)
df_r3 = df_r3.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr3,Yr3,df_r3, 200, zdir='z', offset=-100, cmap='hot')
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
ax.axhline(y=yr3[11], color='r', linestyle = "--")
ax.axvline(x=xr3[16], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\30µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr3, df_r3_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr3), max(xr3), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r3 = y_fit 
peaks, _ = find_peaks(h_r3)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r3, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r3, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("15µmhorizontalr3")
print(max(results_half_plot[0]))
print(" ")

print("15µmhorizontalfullr3")
print(max(results_full_plot[0]))
print(" ")
horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r3 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r3)
ax2.plot(peaks_h, h_r3[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\30µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r3))
horizontal_peaks_position.append(x_new[np.argmax(h_r3)])
master_data_horizontal["30um"] = h_r3

#plot vertical
ax2 = plt.axes()
tck = interpolate.splrep(yr3, df_r3_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr3), max(yr3), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r3 = y_fit 
peaks, _ = find_peaks(v_r3)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r3, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r3, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("30µmverticler3")
print(max(results_half_plot[0]))
print(" ")

print("30µmverticlefullr3")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r3 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r3)
ax2.plot(peaks_v, v_r3[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\30µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r3))
verticle_peaks_position.append(x_new[np.argmax(v_r3)])
master_data_verticle["30um"] = v_r3


df = pd.read_csv(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\2nd march different radius upper 40 50 60 um aligning the ref 3800-3900cnts 4x 55mA mA_0001.ascii.csv",header=None, sep=",")
   
df=df.dropna(axis=1)
print(df)


x = np.linspace(0, 9570, num=320)
x = x/20
y = np.linspace(0, 7650, num=256)
y = y/20
colorbarmax = 4500
colorbartick = 9

X,Y = np.meshgrid(x,y)
df1 = df.to_numpy()
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\radius2_upper_everything_feb2023.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, df1, cmap='hot')
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=13)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=15)
ax.set_zlabel('Photon/s', fontsize=18, fontweight="bold", labelpad=15)


ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.zaxis.label.set_fontsize(18)
ax.zaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_zticklabels(ax.get_zticks(), weight='bold')
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\radius2_upper_everything_3D_feb2023.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

df_r4 = df.iloc[218:242, 206:240]
df_r4_hor = df_r4.iloc[13,:]
df_r4_ver = df_r4.iloc[:, 16]
xr4 = np.linspace(0, 990, num=34)
xr4 = xr4/20
yr4 = np.linspace(0, 690, num=24)
yr4 = yr4/20
colorbarmax = 5000
colorbartick = 9



Xr4,Yr4 = np.meshgrid(xr4,yr4)
df_r4 = df_r4.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr4,Yr4,df_r4, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
ax.axhline(y=yr4[13], color='r', linestyle = "--")
ax.axvline(x=xr4[16], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\40µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr4, df_r4_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr4), max(xr4), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r4 = y_fit 
peaks, _ = find_peaks(h_r4)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r4, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r4, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("40µmhorizontalr4")
print(max(results_half_plot[0]))
print(" ")

print("40µmhorizontalfullr4")
print(max(results_full_plot[0]))
print(" ")

horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r4 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r4)
ax2.plot(peaks_h, h_r4[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")


ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\40µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r4))
horizontal_peaks_position.append(x_new[np.argmax(h_r4)])
master_data_horizontal["40um"] = h_r4

#plot verticle
ax2 = plt.axes()
tck = interpolate.splrep(yr4, df_r4_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr4), max(yr4), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r4 = y_fit 


peaks, _ = find_peaks(v_r4)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r4, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r4, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("40µmverticler4")
print(max(results_half_plot[0]))
print(" ")

print("40µmverticlefullr4")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r4 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r4)
ax2.plot(peaks_v, v_r4[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\40µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r4))
verticle_peaks_position.append(x_new[np.argmax(v_r4)])
master_data_verticle["40um"] = v_r4

df_r5 = df.iloc[81:105, 206:240]
df_r5_hor = df_r5.iloc[11,:]
df_r5_ver = df_r5.iloc[:, 16]
xr5 = np.linspace(0, 990, num=34)
xr5 = xr5/20
yr5 = np.linspace(0, 690, num=24)
yr5 = yr5/20
colorbarmax = 5000
colorbartick = 5

Xr5,Yr5 = np.meshgrid(xr5,yr5)
df_r5 = df_r5.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr5,Yr5,df_r5, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
ax.axhline(y=yr5[11], color='r', linestyle = "--")
ax.axvline(x=xr5[16], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\50µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr5, df_r5_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr5), max(xr5), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r5 = y_fit 
peaks, _ = find_peaks(h_r5)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r5, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r5, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("50µmhorizontalr5")
print(max(results_half_plot[0]))
print(" ")

print("50µmhorizontalfullr5")
print(max(results_full_plot[0]))
print(" ")

horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r5 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r5)
ax2.plot(peaks_h, h_r5[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\50µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r5))
horizontal_peaks_position.append(x_new[np.argmax(h_r5)])
master_data_horizontal["50um"] = h_r5

#plot verticle
ax2 = plt.axes()
tck = interpolate.splrep(yr5, df_r5_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr5), max(yr5), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r5 = y_fit 
peaks, _ = find_peaks(v_r5)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r5, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r5, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("50µmverticler5")
print(max(results_half_plot[0]))
print(" ")

print("50µmverticlefullr5")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r5 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r5)
ax2.plot(peaks_v, v_r5[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\50µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r5))
verticle_peaks_position.append(x_new[np.argmax(v_r5)])
master_data_verticle["50um"] = v_r5

df_r6 = df.iloc[12:36, 206:240]
df_r6_hor = df_r6.iloc[11,:]
df_r6_ver = df_r6.iloc[:, 16]
xr6 = np.linspace(0, 990, num=34)
xr6 = xr6/20
yr6 = np.linspace(0, 690, num=24)
yr6 = yr6/20
colorbarmax = 5000
colorbartick = 5

Xr6,Yr6 = np.meshgrid(xr6,yr6)
df_r6 = df_r6.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr6,Yr6,df_r6, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
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
ax.axhline(y=yr6[11], color='r', linestyle = "--")
ax.axvline(x=xr6[16], color='g', linestyle = "--")
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\60µm_2D_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#plot horizontal
ax2 = plt.axes()
tck = interpolate.splrep(xr6, df_r6_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(xr6), max(xr6), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
h_r6 = y_fit 
peaks, _ = find_peaks(h_r6)
peaks_h = x_new[peaks]
horizontal_peaks.append(peaks_h)

results_half = peak_widths(h_r6, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(h_r6, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("60µmhorizontalr6")
print(max(results_half_plot[0]))
print(" ")

print("60µmhorizontalfullr6")
print(max(results_full_plot[0]))
print(" ")

horizontal_half.append(max(results_half_plot[0]))
horizontal_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(h_r6 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
horizontal_width_cut.append(peak_width)


ax2.plot(x_new, h_r6)
ax2.plot(peaks_h, h_r6[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\60µm_horizontal_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_horizontal.append(max(results_full_plot[0]))
half_width_horizontal.append(max(results_half_plot[0]))
horizontal_peaks.append(max(h_r6))
horizontal_peaks_position.append(x_new[np.argmax(h_r6)])
master_data_horizontal["60um"] = h_r6

#plot vertical
ax2 = plt.axes()
tck = interpolate.splrep(yr6, df_r6_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(yr6), max(yr6), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
v_r6 = y_fit 
peaks, _ = find_peaks(v_r6)
peaks_v = x_new[peaks]
verticle_peaks.append(peaks_v)

results_half = peak_widths(v_r6, peaks, rel_height=0.5)
width = results_half[0]
width = [i*(x_new[1] - x_new[0]) for i in width]
width = np.array(width)
height = results_half[1]
x_min = results_half[2]
x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
x_min = np.array(x_min)
x_max = results_half[3]
x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
x_max = np.array(x_max)
results_half_plot = (width, height, x_min, x_max)

results_full = peak_widths(v_r6, peaks, rel_height=0.865)
width_f = results_full[0]
width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
width_f = np.array(width_f)
height_f = results_full[1]
x_min_f = results_full[2]
x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
x_min_f = np.array(x_min_f)
x_max_f = results_full[3]
x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
x_max_f = np.array(x_max_f)
results_full_plot = (width_f, height_f, x_min_f, x_max_f)


print("60µmverticler6")
print(max(results_half_plot[0]))
print(" ")

print("60µmverticlefullr6")
print(max(results_full_plot[0]))
print(" ")

verticle_half.append(max(results_half_plot[0]))
verticle_full.append(max(results_full_plot[0]))

#Determine the cross-section at y = y_line
y_line = cut
delta = 50
x_close = x_new[np.where(np.abs(v_r6 - y_line) < delta)]
peak_width = np.max(x_close) - np.min(x_close)
verticle_width_cut.append(peak_width)


ax2.plot(x_new, v_r6)
ax2.plot(peaks_v, v_r6[peaks], "o")
ax2.hlines(*results_half_plot[1:], color="C2")
ax2.hlines(*results_full_plot[1:], color="C3")
ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")


ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)

ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["Photon/s", "Peaks", "FWHM", "Full Width", str(cut)+" cnt/s"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\60µm_verticle_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

full_width_verticle.append(max(results_full_plot[0]))
half_width_verticle.append(max(results_half_plot[0]))
verticle_peaks.append(max(v_r6))
verticle_peaks_position.append(x_new[np.argmax(v_r6)])
master_data_verticle["60um"] = v_r6


xr1_fit = np.linspace(min(xr1), max(xr1), 1000)
xr2_fit = np.linspace(min(xr2), max(xr2), 1000)
xr3_fit = np.linspace(min(xr3), max(xr3), 1000)
xr4_fit = np.linspace(min(xr4), max(xr4), 1000)
xr5_fit = np.linspace(min(xr5), max(xr5), 1000)
xr6_fit = np.linspace(min(xr6), max(xr6), 1000)
hor = np.linspace(min(xr6), max(xr6), 1000)
master_data_horizontal["hor_15µm"] = h_r1
master_data_horizontal["hor_20µm"] = h_r2
master_data_horizontal["hor_30µm"] = h_r3
master_data_horizontal["hor_40µm"] = h_r4
master_data_horizontal["hor_50µm"] = h_r5
master_data_horizontal["hor_60µm"] = h_r6
master_data_horizontal["x_position"] = hor
ax2 = plt.axes()
ax2.plot(hor, master_data_horizontal["hor_15µm"])
ax2.plot(hor, master_data_horizontal["hor_20µm"])
ax2.plot(hor, master_data_horizontal["hor_30µm"])
ax2.plot(hor, master_data_horizontal["hor_40µm"])
ax2.plot(hor, master_data_horizontal["hor_50µm"])
ax2.plot(hor, master_data_horizontal["hor_60µm"])
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\horizontal_compile_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


yr1_fit = np.linspace(min(yr1), max(yr1), 1000)
yr2_fit = np.linspace(min(yr2), max(yr2), 1000)
yr3_fit = np.linspace(min(yr3), max(yr3), 1000)
yr4_fit = np.linspace(min(yr4), max(yr4), 1000)
yr5_fit = np.linspace(min(yr5), max(yr5), 1000)
yr6_fit = np.linspace(min(yr6), max(yr6), 1000)
ver = np.linspace(min(yr6), max(yr6), 1000)
master_data_verticle["ver_15µm"] = v_r1
master_data_verticle["ver_20µm"] = v_r2
master_data_verticle["ver_30µm"] = v_r3
master_data_verticle["ver_40µm"] = v_r4
master_data_verticle["ver_50µm"] = v_r5
master_data_verticle["ver_60µm"] = v_r6
master_data_verticle["y_position"] = ver

ax2 = plt.axes()
ax2.plot(ver, master_data_verticle["ver_15µm"])
ax2.plot(ver, master_data_verticle["ver_20µm"])
ax2.plot(ver, master_data_verticle["ver_30µm"])
ax2.plot(ver, master_data_verticle["ver_40µm"])
ax2.plot(ver, master_data_verticle["ver_50µm"])
ax2.plot(ver, master_data_verticle["ver_60µm"])
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold'})
plt.savefig(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\verticle_compile_filter_by_max.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(master_data_horizontal)
print(master_data_verticle)

print("r1 is 30µm")
print("r2 is 20µm")
print("r3 is 15µm")
print("r4 is 60µm")
print("r5 is 50µm")
print("r6 is 40µm")

radius = ["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"]
master_data_verticle.to_excel(r'C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\master_data_verticle.xlsx', index=False)
master_data_horizontal.to_excel(r'C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\master_data_horizontal.xlsx', index=False)
peak_height_horizontal = [max(h_r1), max(h_r2), max(h_r3), max(h_r4), max(h_r5), max(h_r6) ]
peak_height_verticle = [max(v_r1), max(v_r2), max(v_r3), max(v_r4), max(v_r5), max(v_r6) ]

print("full_width_horizontal")
print(full_width_horizontal)
print(" ")
print("full_width_verticle")
print(full_width_verticle)
print(" ")
print("half_width_horizontal")
print(half_width_horizontal)
print(" ")
print("half_width_verticle")
print(half_width_verticle)
print(" ")
print("horizontal_peaks_position")
print(horizontal_peaks_position)
print(" ")
print("verticle_peaks_position")
print(verticle_peaks_position)
print(" ")
print("peak_height_horizontal")
print(peak_height_horizontal)
print(" ")
print("peak_height_verticle")
print(peak_height_verticle)
print(" ")
print("horizontal_width_cut")
print(horizontal_width_cut )
print(" ")
print("verticle_width_cut")
print(verticle_width_cut)
print(" ")

X = df_r1, df_r2, df_r3, df_r4, df_r5, df_r6

for x in X:
    print(x.max().max())

exp_filtered_by_width = [20.511997, 12.2996, 13.958, 15.6158, 17.88, 18.5346 ]
exp_no_filter_max_index = [20.511997, 12.2996, 13.958, 14.207003, 17.44541, 18.5346]
simulated_40um = [15.20, 6.89, 16.33, 21.72, 24.16, 26.37]
max_photon_count = [1335, 1852, 2276, 3955, 4006, 2713]
radius = ["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"]
fig, ax2 = plt.subplots()
ax2.scatter(radius, exp_filtered_by_width, marker = "o", s=50)
ax2.scatter(radius, exp_no_filter_max_index, marker = "s", s=50)
ax2.scatter(radius, simulated_40um, marker = "v", s=50)
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(16)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(16)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax2.spines["right"].set_linewidth(2)
ax2.spines["top"].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.set_ylabel("Beam Width (µm)")
ax2.set_xlabel("Radius of Curvature (µm)")
ax2.legend(["Selection by Largest Beam Width", "Selection by Highest Photon/s", "Simulated Beam Width"], loc = 'lower right', prop={'weight': 'bold'})

ax1 = ax2.twinx()
ax1.plot(radius, max_photon_count, color='blue')
ax1.set_ylabel('Peak Photon/s', color = "blue")
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', color = "blue")
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax1.set_xlabel("Radius of Curvature (µm)")
plt.show()
plt.close()

exp_filtered_by_width = [13.958, 12.2996, 15.6158, 17.88, 18.5346, 20.511997]
exp_no_filter_max_index = [13.958, 12.2996, 14.207003, 17.44541, 18.5346, 20.511997]
simulated_40um = [15.20, 6.89, 16.33, 21.72, 24.16, 26.37]
max_photon_count = [1335, 1852, 2276, 3955, 4006, 2713]
radius = ["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"]
fig, ax2 = plt.subplots()
ax2.scatter(radius, exp_filtered_by_width, marker = "o", s=50)
ax2.scatter(radius, exp_no_filter_max_index, marker = "s", s=50)
ax2.scatter(radius, simulated_40um, marker = "v", s=50)
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(16)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(16)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax2.spines["right"].set_linewidth(2)
ax2.spines["top"].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.set_ylabel("Beam Waist (µm)")
ax2.set_xlabel("Radius of Curvature (µm)")
ax2.legend(["Column Selection by Beam Waist", "Column Selection by Intensity", "Simulated Beam Waist"], prop={'weight': 'bold'})

ax1 = ax2.twinx()
ax1.plot(radius, max_photon_count, color='blue')
ax1.set_ylabel('Peak Photon/s', color = "blue")
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', color = "blue")
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax1.set_xlabel("Radius of Curvature (µm)")
plt.show()
plt.close()
