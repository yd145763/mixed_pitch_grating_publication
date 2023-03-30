# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:10:09 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import stats
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths

filename = ["y_0.7umfixedpitch", "y_0.8umfixedpitch", "y_0.9umfixedpitch", "y_1.0umfixedpitch", "y_1.1umfixedpitch", "y_1.2umfixedpitch"]

for f in filename:
    print(" ")
    print(" ")
    print(" ")
    print(f)
    print(" ")
    print(" ")
    print(" ")
    df = pd.read_csv(r"C:\Users\limyu\Google Drive\focusing grating\2D simulation data 13th March\\"+f+".csv",header=None, sep=";")
    df1 = df.iloc[1:, :3775]
    x = np.linspace(0, 200, num = df1.shape[1])
    z = np.linspace(0, 100, num = df1.shape[0])
    colorbarmax = df1.max().max()
    X,Z = np.meshgrid(x,z)
    fig = plt.figure(figsize=(13, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Z,df1, 200, zdir='z', offset=-100, cmap='jet')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
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
    
    cut_off = 120
    

    max_e_field_list = []
    z1 = np.linspace(0, 100, num = 600)
    for i in range(df1.shape[0]):
        e_field = df1.iloc[i, :]

        index = np.where(e_field == max(e_field))
        index = int(index[0])
        
        max_e_field_list.append(x[index])
    slope, intercept, r_value, p_value, std_err = stats.linregress(max_e_field_list, z1)
    def linear(x):
        y = slope*x+intercept
        return y
    max_e_field_list = np.array(max_e_field_list)
    fit_z = linear(max_e_field_list)
    index_z = np.abs(fit_z - 100).argmin()
    fit_z[:cut_off] = np.nan
    fit_z[index_z:] = np.nan
    max_e_field_list[:cut_off] = np.nan
    max_e_field_list[index_z:] = np.nan
    z1[:cut_off] = np.nan
    z1[index_z:] = np.nan
    plt.scatter(max_e_field_list, z1, s=1)
    plt.plot(max_e_field_list, fit_z, linestyle="--", color = "w")
    plt.show()
    plt.close()
    
    

    df1 = df.iloc[1:, :3775]
    x = np.linspace(0, 200, num = df1.shape[1])
    z = np.linspace(0, 100, num = df1.shape[0])
    colorbarmax = df1.max().max()
    X,Z = np.meshgrid(x,z)
    fig = plt.figure(figsize=(13, 4))
    ax = plt.axes()
    ax.plot(max_e_field_list, fit_z, linestyle="--", color = "w")
    cp=ax.contourf(X,Z,df1, 200, zdir='z', offset=-100, cmap='jet')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
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
    angle = np.arctan(slope)
    angle_deg = np.degrees(angle)
    angle_from_verticle = 90 - angle_deg
    print(" ")
    print(" ")
    print(" ")
    print(angle_from_verticle)
    print(" ")
    print(" ")
    print(" ")   