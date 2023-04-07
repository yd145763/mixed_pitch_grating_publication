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
from sklearn.metrics import mean_squared_error

def gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

master_data_verticle = pd.DataFrame([])
master_data_horizontal = pd.DataFrame([])

horizontal_peaks_position = []
verticle_peaks_position = []

verticle_mse = []
verticle_peaks = []
verticle_half = []
verticle_full = []

horizontal_mse = []
horizontal_peaks = []
horizontal_half = []
horizontal_full = []

filename = []
max_field = []

ix1 = 12
iy1 = 20
ix2 = 12
iy2 = 20
ix3 = 13
iy3 = 20
ix4 = 11
iy4 = 16
ix5 = 13
iy5 = 15
ix6 = 12
iy6 = 16

url = "https://raw.githubusercontent.com/yd145763/mixed_pitch_grating_publication/main/different%20mixture%204%20beams%20near%20ref%20and%20a%20beam%203800-3900cnts%204x%2090mA.csv"
df = pd.read_csv("https://raw.githubusercontent.com/yd145763/mixed_pitch_grating_publication/main/different%20mixture%204%20beams%20near%20ref%20and%20a%20beam%203800-3900cnts%204x%2090mA.csv")




x = np.linspace(0, 9570, num=df.shape[1])
x = x/20
y = np.linspace(0, 7650, num=df.shape[0])
y = y/20
colorbarmax = df.max().max()

colorbartick = 9

X,Y = np.meshgrid(x,y)
df1 = df.to_numpy()

#contour plot for multiple beams
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
plt.show()
plt.close()

#3D plot for multiple beams
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, df1, cmap='jet')
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
plt.show()
plt.close()

df_r1 = df.iloc[228:252, 210:244]
df_r1 = df_r1.reset_index(drop=True)
df_r1.columns = range(df_r1.shape[1])
df_r1_hor = df_r1.iloc[ix1,:]
df_r1_hor = df_r1_hor - min(df_r1_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r1.max().max()
colorbartick = 9
max_field.append(df_r1.iloc[ix1, iy1])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r1 = df_r1.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r1, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix1], color='r', linestyle = "--")
ax.axvline(x=x1[iy1], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r1_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r1_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r1_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r1_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r1_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r1_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r1_hor)
ax2.plot(x1[peaks], df_r1_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r1_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r1_hor)])
master_data_horizontal["grating12_11pitch2_8"] = df_r1_hor
filename.append("grating12_11pitch2_8")

df_r1 = pd.DataFrame(df_r1)
df_r1_hor = df_r1.iloc[ix1,:]
df_r1_hor = df_r1_hor - min(df_r1_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r1_hor)
B = 1/(waist_h**2)
max_df_r1_hor_index = int(np.where(df_r1_hor == max(df_r1_hor))[0])
max_df_r1_hor_x = x1[max_df_r1_hor_index]
distance_x = x1 - max_df_r1_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r1_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r1_hor, fit_y)
horizontal_mse.append(mse)

print("r1 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r1_ver = df_r1.iloc[:, iy1]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r1_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r1_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r1_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r1_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r1_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r1_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r1_ver)
ax2.plot(y1[peaks], df_r1_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r1_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r1_ver)])
master_data_verticle["grating12_11pitch2_8"] = df_r1_ver

df_r1 = pd.DataFrame(df_r1)
df_r1_ver = df_r1.iloc[:,iy1]
df_r1_ver = df_r1_ver - min(df_r1_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for horizontal e-field

A = max(df_r1_ver)
B = 1/(waist_v**2)
max_df_r1_ver_index = int(np.where(df_r1_ver == max(df_r1_ver))[0])
max_df_r1_ver_x = y1[max_df_r1_ver_index]
distance_y = y1 - max_df_r1_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r1_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r1_ver, fit_y)
verticle_mse.append(mse)
print("r1 vertical MSE", mse)


#------------------------------------------------------sample splitter------------------------------------------------------



df_r2 = df.iloc[158:182, 210:244]
df_r2 = df_r2.reset_index(drop=True)
df_r2.columns = range(df_r2.shape[1])
df_r2_hor = df_r2.iloc[ix2,:]
df_r2_hor = df_r2_hor - min(df_r2_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r2.max().max()
colorbartick = 9
max_field.append(df_r2.iloc[ix2, iy2])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r2 = df_r2.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r2, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix2], color='r', linestyle = "--")
ax.axvline(x=x1[iy2], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r2_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r2_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r2_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r2_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r2_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r2_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r2_hor)
ax2.plot(x1[peaks], df_r2_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r2_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r2_hor)])
master_data_horizontal["grating12_100"] = df_r2_hor
filename.append("grating12_100")

df_r2 = pd.DataFrame(df_r2)
df_r2_hor = df_r2.iloc[ix2,:]
df_r2_hor = df_r2_hor - min(df_r2_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r2_hor)
B = 1/(waist_h**2)
max_df_r2_hor_index = int(np.where(df_r2_hor == max(df_r2_hor))[0])
max_df_r2_hor_x = x1[max_df_r2_hor_index]
distance_x = x1 - max_df_r2_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r2_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r2_hor, fit_y)
horizontal_mse.append(mse)
print("r2 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r2_ver = df_r2.iloc[:, iy2]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r2_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r2_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r2_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r2_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r2_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r2_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r2_ver)
ax2.plot(y1[peaks], df_r2_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r2_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r2_ver)])
master_data_verticle["grating12_100"] = df_r2_ver

df_r2 = pd.DataFrame(df_r2)
df_r2_ver = df_r2.iloc[:,iy2]
df_r2_ver = df_r2_ver - min(df_r2_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for vertical e-field

A = max(df_r2_ver)
B = 1/(waist_v**2)
max_df_r2_ver_index = int(np.where(df_r2_ver == max(df_r2_ver))[0])
max_df_r2_ver_x = y1[max_df_r2_ver_index]
distance_y = y1 - max_df_r2_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r2_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r2_ver, fit_y)
verticle_mse.append(mse)
print("r2 vertical MSE", mse)

#------------------------------------------------------sample splitter------------------------------------------------------





url = "https://raw.githubusercontent.com/yd145763/mixed_pitch_grating_publication/main/different%20mixture%204%20beams%20further%20from%20ref%20and%20a%20beam%203800-3900cnts%204x%2090mA.csv"
df = pd.read_csv(r"C:\Users\limyu\Google Drive\different mixture\different mixture 4 beams further from ref and a beam 3800-3900cnts 4x 90mA.csv",header=None, sep=",")
   
df=df.dropna(axis=1)
print(df)


x = np.linspace(0, 9570, num=320)
x = x/20
y = np.linspace(0, 7650, num=256)
y = y/20
colorbarmax = df.max().max()
colorbartick = 9


X,Y = np.meshgrid(x,y)
df1 = df.to_numpy()
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
plt.show()
plt.close()

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface(X, Y, df1, cmap='jet')
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

plt.show()
plt.close()


df_r3 = df.iloc[223:247, 217:251]
df_r3 = df_r3.reset_index(drop=True)
df_r3.columns = range(df_r3.shape[1])
df_r3_hor = df_r3.iloc[ix3,:]
df_r3_hor = df_r3_hor - min(df_r3_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r3.max().max()
colorbartick = 9
max_field.append(df_r3.iloc[ix3, iy3])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r3 = df_r3.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r3, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix3], color='r', linestyle = "--")
ax.axvline(x=x1[iy3], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r3_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r3_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r3_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r3_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r3_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r3_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r3_hor)
ax2.plot(x1[peaks], df_r3_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r3_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r3_hor)])
master_data_horizontal["grating12_11pitch8_2"] = df_r3_hor
filename.append("grating12_11pitch8_2")

df_r3 = pd.DataFrame(df_r3)
df_r3_hor = df_r3.iloc[ix3,:]
df_r3_hor = df_r3_hor - min(df_r3_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r3_hor)
B = 1/(waist_h**2)
max_df_r3_hor_index = int(np.where(df_r3_hor == max(df_r3_hor))[0])
max_df_r3_hor_x = x1[max_df_r3_hor_index]
distance_x = x1 - max_df_r3_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r3_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r3_hor, fit_y)
horizontal_mse.append(mse)
print("r3 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r3_ver = df_r3.iloc[:, iy3]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r3_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r3_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r3_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r3_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r3_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r3_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r3_ver)
ax2.plot(y1[peaks], df_r3_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r3_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r3_ver)])
master_data_verticle["grating12_11pitch8_2"] = df_r3_ver

df_r3 = pd.DataFrame(df_r3)
df_r3_ver = df_r3.iloc[:,iy3]
df_r3_ver = df_r3_ver - min(df_r3_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for vertical e-field

A = max(df_r3_ver)
B = 1/(waist_v**2)
max_df_r3_ver_index = int(np.where(df_r3_ver == max(df_r3_ver))[0])
max_df_r3_ver_x = y1[max_df_r3_ver_index]
distance_y = y1 - max_df_r3_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r3_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r3_ver, fit_y)
verticle_mse.append(mse)
print("r3 vertical MSE", mse)

#------------------------------------------------------sample splitter------------------------------------------------------



df_r4 = df.iloc[155:179, 217:251]
df_r4 = df_r4.reset_index(drop=True)
df_r4.columns = range(df_r4.shape[1])
df_r4_hor = df_r4.iloc[ix4,:]
df_r4_hor = df_r4_hor - min(df_r4_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r4.max().max()
colorbartick = 9
max_field.append(df_r4.iloc[ix4, iy4])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r4 = df_r4.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r4, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix4], color='r', linestyle = "--")
ax.axvline(x=x1[iy4], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r4_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r4_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r4_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r4_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r4_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r4_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r4_hor)
ax2.plot(x1[peaks], df_r4_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r4_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r4_hor)])
master_data_horizontal["grating12_11pitch6_4"] = df_r4_hor
filename.append("grating12_11pitch6_4")

df_r4 = pd.DataFrame(df_r4)
df_r4_hor = df_r4.iloc[ix4,:]
df_r4_hor = df_r4_hor - min(df_r4_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r4_hor)
B = 1/(waist_h**2)
max_df_r4_hor_index = int(np.where(df_r4_hor == max(df_r4_hor))[0])
max_df_r4_hor_x = x1[max_df_r4_hor_index]
distance_x = x1 - max_df_r4_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r4_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r4_hor, fit_y)
horizontal_mse.append(mse)
print("r4 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r4_ver = df_r4.iloc[:, iy4]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r4_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r4_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r4_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r4_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r4_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r4_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r4_ver)
ax2.plot(y1[peaks], df_r4_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r4_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r4_ver)])
master_data_verticle["grating12_11pitch6_4"] = df_r4_ver

df_r4 = pd.DataFrame(df_r4)
df_r4_ver = df_r4.iloc[:,iy4]
df_r4_ver = df_r4_ver - min(df_r4_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for vertical e-field

A = max(df_r4_ver)
B = 1/(waist_v**2)
max_df_r4_ver_index = int(np.where(df_r4_ver == max(df_r4_ver))[0])
max_df_r4_ver_x = y1[max_df_r4_ver_index]
distance_y = y1 - max_df_r4_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r4_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r4_ver, fit_y)
verticle_mse.append(mse)
print("r4 vertical MSE", mse)

#------------------------------------------------------sample splitter------------------------------------------------------



df_r5 = df.iloc[83:107, 217:251]
df_r5 = df_r5.reset_index(drop=True)
df_r5.columns = range(df_r5.shape[1])
df_r5_hor = df_r5.iloc[ix5,:]
df_r5_hor = df_r5_hor - min(df_r5_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r5.max().max()
colorbartick = 9
max_field.append(df_r1.iloc[ix5, iy5])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r5 = df_r5.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r5, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix5], color='r', linestyle = "--")
ax.axvline(x=x1[iy5], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r5_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r5_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r5_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r5_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r5_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r5_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r5_hor)
ax2.plot(x1[peaks], df_r5_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r5_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r5_hor)])
master_data_horizontal["grating12_11pitch5_5"] = df_r5_hor
filename.append("grating12_11pitch5_5")

df_r5 = pd.DataFrame(df_r5)
df_r5_hor = df_r5.iloc[ix5,:]
df_r5_hor = df_r5_hor - min(df_r5_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r5_hor)
B = 1/(waist_h**2)
max_df_r5_hor_index = int(np.where(df_r5_hor == max(df_r5_hor))[0])
max_df_r5_hor_x = x1[max_df_r5_hor_index]
distance_x = x1 - max_df_r5_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r5_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r5_hor, fit_y)
horizontal_mse.append(mse)
print("r5 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r5_ver = df_r5.iloc[:, iy5]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r5_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r5_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r5_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r5_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r5_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r5_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r5_ver)
ax2.plot(y1[peaks], df_r5_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r5_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r5_ver)])
master_data_verticle["grating12_11pitch5_5"] = df_r5_ver

df_r5 = pd.DataFrame(df_r5)
df_r5_ver = df_r5.iloc[:,iy5]
df_r5_ver = df_r5_ver - min(df_r5_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for vertical e-field

A = max(df_r5_ver)
B = 1/(waist_v**2)
max_df_r5_ver_index = int(np.where(df_r5_ver == max(df_r5_ver))[0])
max_df_r5_ver_x = y1[max_df_r5_ver_index]
distance_y = y1 - max_df_r5_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r5_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r5_ver, fit_y)
verticle_mse.append(mse)
print("r5 vertical MSE", mse)

#------------------------------------------------------sample splitter------------------------------------------------------



df_r6 = df.iloc[14:38, 214:248]
df_r6 = df_r6.reset_index(drop=True)
df_r6.columns = range(df_r6.shape[1])
df_r6_hor = df_r6.iloc[ix6,:]
df_r6_hor = df_r6_hor - min(df_r6_hor)

x1 = np.linspace(0, 990, num=34)
x1 = x1/20
y1 = np.linspace(0, 690, num=24)
y1 = y1/20
colorbarmax = df_r6.max().max()
colorbartick = 9
max_field.append(df_r1.iloc[ix6, iy6])

#contour plot for single beam
X1,Y1 = np.meshgrid(x1,y1)
df_r6 = df_r6.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(X1,Y1,df_r6, 200, zdir='z', offset=-100, cmap='jet')
clb=fig.colorbar(cp, ticks=(np.linspace(0, colorbarmax, num = 6)).tolist())
clb.ax.set_title('cnt/s', fontweight="bold")
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
ax.axhline(y=y1[ix6], color='r', linestyle = "--")
ax.axvline(x=x1[iy6], color='g', linestyle = "--")
plt.show()
plt.close()

tck = interpolate.splrep(x1, df_r6_hor.tolist(), s=2, k=4) 
x_new = np.linspace(min(x1), max(x1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r6_hor = y_fit
x1 = x_new


#calculate peaks for horizontal line
peaks, _ = find_peaks(df_r6_hor)
peaks_h = x1[peaks]
peaks_height_h = df_r6_hor[peaks]
max_index_h = np.where(peaks_height_h == max(peaks_height_h))
max_index_h = int(max_index_h[0])
results_half = peak_widths(df_r6_hor, peaks, rel_height=0.5)

#convert peaks from index to x
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_h = FWHM_array[max_index_h]


results_full = peak_widths(df_r6_hor, peaks, rel_height=0.865)

#convert peaks from index to x
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = x1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = x1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_h = waist_array[max_index_h]




horizontal_half.append(FWHM_h)
horizontal_full.append(waist_h)

ax2 = plt.axes()

ax2.plot(x1, df_r6_hor)
ax2.plot(x1[peaks], df_r6_hor[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r6_hor))
horizontal_peaks_position.append(x1[np.argmax(df_r6_hor)])
master_data_horizontal["grating12_11pitch4_6"] = df_r6_hor
filename.append("grating12_11pitch4_6")

df_r6 = pd.DataFrame(df_r6)
df_r6_hor = df_r6.iloc[ix6,:]
df_r6_hor = df_r6_hor - min(df_r6_hor)


x1 = np.linspace(0, 990, num=34)
x1 = x1/20


#gaussian fit for horizontal e-field

A = max(df_r6_hor)
B = 1/(waist_h**2)
max_df_r6_hor_index = int(np.where(df_r6_hor == max(df_r6_hor))[0])
max_df_r6_hor_x = x1[max_df_r6_hor_index]
distance_x = x1 - max_df_r6_hor_x
fit_y = gauss(distance_x, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_x, df_r6_hor, s = 5, color = "black")
ax.plot(distance_x, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (x-axis, µm)")
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r6_hor, fit_y)
horizontal_mse.append(mse)
print("r6 horizontal MSE", mse)

#------------------------------------------------------xy splitter------------------------------------------------------


df_r6_ver = df_r6.iloc[:, iy6]
y1 = np.linspace(0, 690, num=24)
y1 = y1/20

tck = interpolate.splrep(y1, df_r6_ver.tolist(), s=2, k=4) 
x_new = np.linspace(min(y1), max(y1), 1000)
y_fit = interpolate.BSpline(*tck)(x_new)
df_r6_ver = y_fit
y1 = x_new

#calculate peaks for vertical line
peaks, _ = find_peaks(df_r6_ver)
peaks_v = y1[peaks]
peaks_height_v = df_r6_ver[peaks]
max_index_v = np.where(peaks_height_v == max(peaks_height_v))
max_index_v = int(max_index_v[0])
results_half = peak_widths(df_r6_ver, peaks, rel_height=0.5)

#convert peaks from index to y
height_plot = results_half[1]
x_min = results_half[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_half[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_half[0]
width_plot = x_max_plot - x_min_plot
results_half_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

FWHM_array = results_half_plot[0]
FWHM_v = FWHM_array[max_index_v]


results_full = peak_widths(df_r6_ver, peaks, rel_height=0.865)

#convert peaks from index to y
height_plot = results_full[1]
x_min = results_full[2]
x_min_int = x_min.astype(int)
x_min_plot = y1[x_min_int]
x_max = results_full[3]
x_max_int = x_max.astype(int)
x_max_plot = y1[x_max_int]
width = results_full[0]
width_plot = x_max_plot - x_min_plot
results_full_plot = (width_plot, height_plot, x_min_plot, x_max_plot)

waist_array = results_full_plot[0]
waist_v = waist_array[max_index_v]

verticle_half.append(FWHM_v)
verticle_full.append(waist_v)


ax2 = plt.axes()

ax2.plot(y1, df_r6_ver)
ax2.plot(y1[peaks], df_r6_ver[peaks], "o")
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
plt.ylabel("Photon count (cnt/s)")
plt.legend(["Photon count (cnt/s)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
plt.show()
plt.close()



horizontal_peaks.append(max(df_r6_ver))
horizontal_peaks_position.append(y1[np.argmax(df_r6_ver)])
master_data_verticle["grating12_11pitch4_6"] = df_r6_ver

df_r6 = pd.DataFrame(df_r6)
df_r6_ver = df_r6.iloc[:,iy6]
df_r6_ver = df_r6_ver - min(df_r6_ver)


y1 = np.linspace(0, 690, num=24)
y1 = y1/20


#gaussian fit for vertical e-field

A = max(df_r6_ver)
B = 1/(waist_v**2)
max_df_r6_ver_index = int(np.where(df_r6_ver == max(df_r6_ver))[0])
max_df_r6_ver_x = y1[max_df_r6_ver_index]
distance_y = y1 - max_df_r6_ver_x
fit_y = gauss(distance_y, A, B)



#initial the graph and plot e-field, beam waist, and peaks
fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
ax.scatter(distance_y, df_r6_ver, s = 5, color = "black")
ax.plot(distance_y, fit_y, color = "red")


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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Distance from peak E-field (y-axis, µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["E-field (eV)", "Gaussian Fit"], prop={'weight': 'bold','size': 10})
plt.show()
plt.close()

mse = mean_squared_error(df_r6_ver, fit_y)
verticle_mse.append(mse)
print("r6 vertical MSE", mse)

#------------------------------------------------------sample splitter------------------------------------------------------

#initial the graph
x_plot = np.linspace(min(x1), max(x1), 1000)

grating = ['grating12_100', 'grating12_11pitch8_2',
       'grating12_11pitch6_4', 'grating12_11pitch5_5', 'grating12_11pitch4_6', 'grating12_11pitch2_8']
grating_label = ["10(1.2µm)", "8(1.2µm) : 2(1.1µm)", "6(1.2µm) : 4(1.1µm)", "5(1.2µm) : 5(1.1µm)", "4(1.2µm) : 6(1.1µm)", "2(1.2µm) : 8(1.1µm)"]
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
for g in grating:
    ax.plot(x_plot, master_data_horizontal[g])
    
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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon count (cnt/s)")
plt.legend(grating_label, prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.show()
plt.close()

#------------------------------------------------------xy splitter------------------------------------------------------

#initial the graph
y_plot = np.linspace(min(y1), max(y1), 1000)

grating = ['grating12_100', 'grating12_11pitch8_2',
       'grating12_11pitch6_4', 'grating12_11pitch5_5', 'grating12_11pitch4_6', 'grating12_11pitch2_8']
grating_label = ["10(1.2µm)", "8(1.2µm) : 2(1.1µm)", "6(1.2µm) : 4(1.1µm)", "5(1.2µm) : 5(1.1µm)", "4(1.2µm) : 6(1.1µm)", "2(1.2µm) : 8(1.1µm)"]
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
for g in grating:
    ax.plot(y_plot, master_data_verticle[g])
    
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
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon count (cnt/s)")
plt.legend(grating_label, prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.show()
plt.close()

print("horizontal_mse")
print(horizontal_mse)
print(" ")
print("horizontal_full")
print(horizontal_full)
print(" ")
print("filename")
print(filename)
print(" ")
print("max_field")
print(max_field )
print(" ")
extracted_data_df = pd.DataFrame()
extracted_data_df["filename"] = filename
extracted_data_df["horizontal_full"] = horizontal_full
extracted_data_df["horizontal_half"] = horizontal_half

extracted_data_df["horizontal_mse"] = horizontal_mse
extracted_data_df["max_field"] = max_field

extracted_data_df["filename"] = filename
extracted_data_df["verticle_full"] = verticle_full
extracted_data_df["verticle_half"] = verticle_half

extracted_data_df["verticle_mse"] = verticle_mse
extracted_data_df["max_field"] = max_field

print(extracted_data_df["filename"])
extracted_data_df = extracted_data_df.reindex([1,2,3,4,5,0])
print(extracted_data_df["filename"])

extracted_data_df.to_csv(r"C:\Users\limyu\Google Drive\different mixture\extracted_data_df.csv", index=False)
