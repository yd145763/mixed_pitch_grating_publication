# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:55:20 2023

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




def wholegraph(url):

    df = pd.read_csv(url, sep=",")

    
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

def everything(horline1, horline2, verline1, verline2, url, column_name):
    df = pd.read_csv(url, sep=",")
    df_r = df.iloc[horline1:horline2, verline1:verline2]
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
        print("widest row index", row_idx)
        print("widest column index", col_idx)
        print(" ")
    df_r1 = df.iloc[horline1:horline2, verline1:verline2]
    df_r1 = df_r1.reset_index(drop=True)
    df_r1.columns = range(df_r1.shape[1])
    df_r1_hor = df_r1.iloc[row_idx,:]
    df_r1_hor = df_r1_hor - min(df_r1_hor)
    

    
    x1 = np.linspace(0, 990, num=34)
    x1 = x1/20
    y1 = np.linspace(0, 690, num=24)
    y1 = y1/20
    colorbarmax = df_r1.max().max()
    colorbartick = 9
    max_field.append(df_r1.iloc[row_idx, col_idx])
    
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
    ax.axhline(y=y1[row_idx], color='r', linestyle = "--")
    ax.axvline(x=x1[col_idx], color='g', linestyle = "--")
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
    master_data_horizontal[column_name] = df_r1_hor
    filename.append(column_name)
    
    df_r1 = pd.DataFrame(df_r1)
    df_r1_hor = df_r1.iloc[row_idx,:]
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
    
    
    df_r1_ver = df_r1.iloc[:, col_idx]
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
    master_data_verticle[column_name] = df_r1_ver
    
    df_r1 = pd.DataFrame(df_r1)
    df_r1_ver = df_r1.iloc[:,col_idx]
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

url1 = "https://raw.githubusercontent.com/yd145763/mixed_pitch_grating_publication/main/different%20mixture%204%20beams%20near%20ref%20and%20a%20beam%203800-3900cnts%204x%2090mA.csv"
wholegraph(url1)

#r1
everything(228,252,210,244, url1, "grating12_11pitch2_8")

#r2
everything(158,182,210,244, url1, "grating12_100")


url2 = "https://raw.githubusercontent.com/yd145763/mixed_pitch_grating_publication/main/different%20mixture%204%20beams%20further%20from%20ref%20and%20a%20beam%203800-3900cnts%204x%2090mA.csv"
wholegraph(url2)

#r3
everything(222, 246, 216, 250, url2, "grating12_11pitch8_2")

#r4
everything(155, 179, 213, 247, url2, "grating12_11pitch6_4")

#r5
everything(87, 111, 217, 251, url2, "grating12_11pitch5_5")

#r6
everything(14, 38, 214, 248, url2, "grating12_11pitch4_6")

x1 = np.linspace(0, 990, num=34)
x1 = x1/20

y1 = np.linspace(0, 690, num=24)
y1 = y1/20

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