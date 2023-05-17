# -*- coding: utf-8 -*-
"""
Created on Jan 12 12:22:00 2023

@author: J. Rafael Otaíza P.
Proyecto Vibraciones CSO - Modulo 01
"""

import os
from datetime import datetime 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import math
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = '1.1'

def Save_Mod01():
    return

def Load_Syscom(directoryPath, file):
    file_name = os.path.join(directoryPath + file + '.txt')
    f = open(file_name,'r')
    
    line_num = 0
    lines = f.readlines()
    f.close()
    for line in lines:
        line_num += 1
        if lines[line_num][0] != '#' and line_num <= 84:
            data = pd.read_csv(file_name, skiprows=line_num, sep=" ", header=None)
            data.columns = ["Time", "Filter_X", "Filter_Y", "Filter_Z"]
            mic = 0
            break
        elif lines[line_num][0] != '#' and line_num > 84:
            data = pd.read_csv(file_name, skiprows=line_num, sep=" ", header=None)
            data.columns = ["Time", "Filter_X", "Filter_Y", "Filter_Z", "Micro"]
            mic = 1
            break
    
    del lines
    return data, mic

def Load_Seed(directoryPath, file,scale=False):
    file_name = os.path.join(directoryPath + file + '.csv')
    data = pd.read_csv(file_name, skiprows=0, sep=';')
    
    if data.shape[1] == 1:
        del data
        data = pd.read_csv(file_name, skiprows=0, sep=',')
        if data.iloc[0][0] == 0:
            del data
            data = pd.read_csv(file_name, skiprows=0, sep=',', header=None)
            data.columns=["Time", "Filter_X", "Filter_Y", "Filter_Z"]
    
    if data.iloc[0][0] == 0:
        del data
        data = pd.read_csv(file_name, skiprows=0, sep=';', header=None)
        data.columns=["Time", "Filter_X", "Filter_Y", "Filter_Z"]
    else:
        data.columns=["Time", "Filter_X", "Filter_Y", "Filter_Z"]
        
    if scale is True:
        # define min max scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # transform data
        s_tmp = data.iloc[:, 1:].copy()
        s_tmp = pd.DataFrame(scaler.fit_transform(s_tmp),columns=s_tmp.columns)
        data.iloc[:,1:] = s_tmp.iloc[:,0:]
    
    #print(data.iloc[0,0])
    #if data.iloc[0,0] == 'Time':
    #    data = data.drop([0])
    
    data.iloc[:, 0] = data.iloc[:, 0]-data.iloc[0, 0]
    return data
    
def Offset(data_raw, offset_lim=0.1):
    offset = []
    for eje in data_raw.columns[1:]:
        offset_tmp = np.mean(data_raw[eje])
        offset.append(offset_tmp)
        if offset_tmp >= offset_lim:
            data_raw[eje] = data_raw[eje] - offset_tmp            
    
    return data_raw, pd.DataFrame([offset],columns=['offset_X','offset_Y','offset_Z'])

def Seed_Waveform(data,t0,t1,name,scale=False,save=False):
    seed = data[(data['Time']>=t0)&(data['Time']<=t1)].iloc[:,0:4].copy()
    if scale is True:
        # define min max scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # transform data
        s_tmp = seed.iloc[:, 1:].copy()

        s_tmp = pd.DataFrame(scaler.fit_transform(s_tmp),columns=s_tmp.columns)
        seed.iloc[:,1:] = s_tmp.iloc[:,0:]
    if save is True:
        seed.iloc[:,0] = seed.iloc[:,0]-seed.iloc[0,0]
        seed.to_csv(name, index=False, sep=',')

    t_ini = seed.iloc[0,0]
    seed.iloc[:,0] = seed.iloc[:,0]-t_ini
    seed = seed.reset_index(drop=True)
    
    return seed

##def Special_Holes(n,sep,ang,xc,yc,zc):
##    H = []
##    sn = np.sin((ang+90)*np.pi/180)
##    cs = np.cos((ang+90)*np.pi/180)
##    if n%2 != 0:
##        x0 = xc + sep*(n//2)*cs
##        y0 = yc + sep*(n//2)*sn
##    elif n%2 == 0:
##        x0 = xc + sep*(n//2 - 0.5)*cs
##        y0 = yc + sep*(n//2 - 0.5)*sn
##    
##    for i in range(n):
##        H.append([x0-i*sep*cs,y0-i*sep*sn,zc]) 
##   
##    return pd.DataFrame(np.array(H).round(decimals=3),columns=['X','Y','Z'])

def Ideal_Test(seed,n,Delay,absolute):
    amp = []
    seed_tmp = seed.copy()
    seed_tmp['SV'] = np.sqrt(seed_tmp.iloc[:,1]**2 + seed_tmp.iloc[:,2]**2 + seed_tmp.iloc[:,3]**2)  
    div = 2
    mult = 1
    if seed['Time'].iloc[-1] < 0.2:
        div = 1
        mult = 2
    
    spacer2 = np.zeros(((mult*n//div)*seed_tmp.shape[0],6), dtype=float)
    dt = round(seed_tmp.iloc[1,0]-seed_tmp.iloc[0,0],4)
    #dt = seed_tmp.iloc[1,0]-seed_tmp.iloc[0,0]
    spacer2[:,0] = np.arange(0,spacer2.shape[0]*dt,dt)
    #spacer2[:,0] = np.arange(seed_tmp.iloc[0,0],spacer2.shape[0]*dt + seed_tmp.iloc[0,0],dt)
    
    if np.isscalar(Delay) == True:
        d = Delay
        for h in range(0,n):
            #indx = np.where(np.isclose(h*d, spacer2[:,0], rtol=1e-04, atol=1e-03, equal_nan=True))[0][0]
            indx = np.where(np.isclose(h*d, spacer2[:,0], rtol=1e-04, equal_nan=True))[0][0]
            #print(indx, spacer2[indx,0], np.isclose(h*d, spacer2[:,0], rtol=1e-04, equal_nan=True))
            
            #spacer2[:,0] = np.arange(0,spacer2.shape[0]*dt,dt)
            spacer2[indx:(indx+seed_tmp.shape[0]),1:5] = spacer2[indx:(indx+seed_tmp.shape[0]),1:5] + seed_tmp.iloc[:,1 :].to_numpy()
        
        #spacer2[:,4] = np.sqrt(spacer2[:,1]**2 + spacer2[:,2]**2 + spacer2[:,3]**2)  
        spacer2[:,5] = d
        if absolute is False:
            amp = [d, max(spacer2[:,1])/max(seed_tmp.iloc[:,1]), max(spacer2[:,2])/max(seed_tmp.iloc[:,2]), max(spacer2[:,3])/max(seed_tmp.iloc[:,3]), max(spacer2[:,4])/max(seed_tmp.iloc[:,4])]
        else:
            amp = [d, max(spacer2[:,1]), max(spacer2[:,2]), max(spacer2[:,3]), max(spacer2[:,4])]
            
        return pd.DataFrame(spacer2,columns=['Time','Filter_X','Filter_Y','Filter_Z','Filter_SV','Delay']), pd.DataFrame([amp],columns=['Delay','Amp_X','Amp_Y','Amp_Z','Amp_SV'])
    else:
        Vib = pd.DataFrame() 
        
        #print(seed_tmp)
        
        for d in Delay:
            spacer2 = np.zeros(((mult*n//div)*seed_tmp.shape[0],5), dtype=float)
            dt = round(seed_tmp.iloc[1,0]-seed_tmp.iloc[0,0],4)
            #dt = seed_tmp.iloc[1,0]-seed_tmp.iloc[0,0]
            spacer2[:,0] = np.arange(0,spacer2.shape[0]*dt,dt)
            #spacer2[:,0] = np.arange(seed_tmp.iloc[0,0],spacer2.shape[0]*dt + seed_tmp.iloc[0,0],dt)
            
            for h in range(0,n):
                #indx = np.where(np.isclose(h*delay, spacer2[:,0], rtol=1e-04, atol=1e-03, equal_nan=True))[0][0]
                indx = np.where(np.isclose(h*d, spacer2[:,0], rtol=1e-04, equal_nan=True))[0][0]
                #spacer2[indx:(indx+seed_tmp.shape[0]),1:4] = spacer2[indx:(indx+seed_tmp.shape[0]),1:4] + seed_tmp.iloc[:,1 :].to_numpy()
                
                spacer2[indx:(indx+seed_tmp.shape[0]),1:5] = spacer2[indx:(indx+seed_tmp.shape[0]),1:5] + seed_tmp.iloc[:,1:5].to_numpy()
            
            #spacer2[:,4] = np.sqrt(spacer2[:,1]**2 + spacer2[:,2]**2 + spacer2[:,3]**2)
            vib = pd.DataFrame(spacer2,columns=['Time','Filter_X','Filter_Y','Filter_Z','Filter_SV'])
            vib['Delay'] = d
            Vib = pd.concat([Vib,vib])
            
            if absolute is False:
                amp_tmp = [d, max(spacer2[:,1])/max(seed_tmp.iloc[:,1]), max(spacer2[:,2])/max(seed_tmp.iloc[:,2]), max(spacer2[:,3])/max(seed_tmp.iloc[:,3]), max(spacer2[:,4])/max(seed_tmp.iloc[:,4])]
            else:
                amp_tmp = [d, max(spacer2[:,1]), max(spacer2[:,2]), max(spacer2[:,3]), max(spacer2[:,4])]

            amp.append(amp_tmp)

        del spacer2, vib
        
        #print(Vib)
        #Vib = Vib.reset_index(drop=True)
        return Vib, pd.DataFrame(amp,columns=['Delay','Amp_X','Amp_Y','Amp_Z','Amp_SV'])

def Special_ppvFreq(data, hei_rop=0.1, dis_rop=25):
    ### PEAK FINDING ALGORITHM ###
    min_t = min(data["Time"])
    avg_z = np.mean(data["Filter_Z"])
    data["Time"] = data["Time"] - min_t

    peaks_xp, _ = find_peaks(data["Filter_X"], height=hei_rop, threshold = None, distance=dis_rop)
    peaks_xm, _ = find_peaks(-data["Filter_X"], height=hei_rop, threshold = None, distance=dis_rop)
    zc_x = np.where(np.diff(np.sign(data["Filter_X"])))[0]

    peaks_yp, _ = find_peaks(data["Filter_Y"], height=hei_rop, threshold = None, distance=dis_rop)
    peaks_ym, _ = find_peaks(-data["Filter_Y"], height=hei_rop, threshold = None, distance=dis_rop)
    zc_y = np.where(np.diff(np.sign(data["Filter_Y"])))[0]

    peaks_zp, _ = find_peaks(data["Filter_Z"], height=hei_rop, threshold = None, distance=dis_rop)
    peaks_zm, _ = find_peaks(-data["Filter_Z"], height=hei_rop, threshold = None, distance=dis_rop)
    zc_z = np.where(np.diff(np.sign(data["Filter_Z"])))[0]
    if zc_z.size == 0 or peaks_zm.size == 0:
        data["Filter_Z"] = data["Filter_Z"] - np.mean(data["Filter_Z"])
        peaks_zm, _ = find_peaks(-data["Filter_Z"], height=hei_rop, threshold = None, distance=dis_rop)
        zc_z = np.where(np.diff(np.sign(data["Filter_Z"])))[0]
        
    freq_x = []
    for i in np.concatenate((peaks_xp,peaks_xm)):
        R = data["Time"][min(zc_x[zc_x > i], default=0)]
        L = data["Time"][max(zc_x[zc_x < i], default=0)]
        freq_x.append(1/(2*(R-L)))

    freq_y = []
    for i in np.concatenate((peaks_yp,peaks_ym)):
        R = data['Time'][min(zc_y[zc_y > i], default=0)]
        L = data['Time'][max(zc_y[zc_y < i], default=0)]
        freq_y.append(1/(2*(R-L)))

    freq_z = []
    for i in np.concatenate((peaks_zp,peaks_zm)):
        R = data['Time'][min(zc_z[zc_z > i], default=0)]
        L = data['Time'][max(zc_z[zc_z < i], default=0)]
        freq_z.append(1/(2*(R-L)))

    data["Filter_Z"] = data["Filter_Z"] + avg_z

    peak_out_x = np.column_stack((freq_x, abs(data["Filter_X"][np.concatenate((peaks_xp,peaks_xm))])))
    peak_out_y = np.column_stack((freq_y, abs(data["Filter_Y"][np.concatenate((peaks_yp,peaks_ym))])))
    peak_out_z = np.column_stack((freq_z, abs(data["Filter_Z"][np.concatenate((peaks_zp,peaks_zm))])))

    peak_out_x = pd.DataFrame(peak_out_x)
    peak_out_x.columns = ["Frequency", "peak_X"]
    index_x = peak_out_x[peak_out_x["Frequency"] < 0].index
    peak_out_x.drop(index_x , inplace=True)
    peak_out_x.reset_index(drop=True)

    peak_out_y = pd.DataFrame(peak_out_y)
    peak_out_y.columns = ["Frequency", "peak_Y"]
    index_y = peak_out_y[peak_out_y["Frequency"] < 0].index
    peak_out_y.drop(index_y , inplace=True)
    peak_out_y.reset_index(drop=True)

    peak_out_z = pd.DataFrame(peak_out_z)
    peak_out_z.columns = ["Frequency", "peak_Z"]
    index_z = peak_out_z[peak_out_z["Frequency"] < 0].index
    peak_out_z.drop(index_z , inplace=True)
    peak_out_z.reset_index(drop=True)
    
    return peak_out_x, peak_out_y, peak_out_z
    
    ### END OF PEAK FINDING ###

def Special_DomFreq(seed,n,Delay):
    vib, amp = Ideal_Test(seed,n,Delay,absolute=True)
  
    fft_rop = []
    
    if np.isscalar(Delay) == True:
        N = vib.shape[0] #round(max(seed_cso['Time'])*1000)
        T = 5e-4
        xf = fftfreq(N, T)[:N//2]
    
        yf_x = 1.0/N * np.abs(fft(vib['Filter_X'].values))
        yf_y = 1.0/N * np.abs(fft(vib['Filter_Y'].values))
        yf_z = 1.0/N * np.abs(fft(vib['Filter_Z'].values))
        
        indx = np.argmax(yf_x)
        indy = np.argmax(yf_y)
        indz = np.argmax(yf_z)
        
        fft_rop = [Delay, xf[indx], xf[indy], xf[indz]]
        return pd.DataFrame([fft_rop],columns=['Delay','FreqDom_X','FreqDom_Y','FreqDom_Z'])
    else:
        for d in Delay:
            N = vib[vib['Delay']==d].shape[0] #round(max(seed_cso['Time'])*1000)
            T = round(vib[vib['Delay']==d]['Time'][2]-vib[vib['Delay']==d]['Time'][1],4)#1e-3
            xf = fftfreq(N, T)[:N//2]
            
            yf_x = 1.0/N * np.abs(fft(vib[vib['Delay']==d]['Filter_X'].values))
            yf_y = 1.0/N * np.abs(fft(vib[vib['Delay']==d]['Filter_Y'].values))
            yf_z = 1.0/N * np.abs(fft(vib[vib['Delay']==d]['Filter_Z'].values))

            indx = np.argmax(yf_x)
            indy = np.argmax(yf_y)
            indz = np.argmax(yf_z)

            fft_tmp = [d, xf[indx], xf[indy], xf[indz]]
            fft_rop.append(fft_tmp)
            
            del yf_x, yf_y, yf_z, indx, indy, indz, fft_tmp
            
        return pd.DataFrame(fft_rop,columns=['Delay','FreqDom_X','FreqDom_Y','FreqDom_Z'])

def Special_Acel(seed,n,Delay):
    vib, amp = Ideal_Test(seed,n,Delay,absolute=True)
    
    if np.isscalar(Delay) == True:
        x = vib2['Time'].values
        y1 = vib2['Filter_X'].values
        y2 = vib2['Filter_Y'].values
        y3 = vib2['Filter_Z'].values

        dy1 = np.zeros(y1.shape,np.float)
        dy1[0:-1] = np.diff(y1)/np.diff(x)
        dy1[-1] = (y1[-1] - y1[-2])/(x[-1] - x[-2])
            
        dy2 = np.zeros(y2.shape,np.float)
        dy2[0:-1] = np.diff(y2)/np.diff(x)
        dy2[-1] = (y2[-1] - y2[-2])/(x[-1] - x[-2])
            
        dy3 = np.zeros(y3.shape,np.float)
        dy3[0:-1] = np.diff(y3)/np.diff(x)
        dy3[-1] = (y3[-1] - y3[-2])/(x[-1] - x[-2])
            
        dysv = np.sqrt(dy1**2 + dy2**2 + dy3**2)
            
        acel_tmp = [d, max(dy1)/(9.8), max(dy2)/(9.8), max(dy3)/(9.8), 
                    np.sqrt(max(dy1)**2 + max(dy2)**2 + max(dy3)**2)/(9.8)]
                    
        return pd.DataFrame([acel_tmp],columns=['Delay','Acel_X','Acel_Y','Acel_Z','Acel_SV'])
    else:
        acel = []

        for d in Delay:
            vib2 = vib[vib['Delay']==d].iloc[:,0:4].copy()
            vib2 = vib2.reset_index(drop=True)

            x = vib2['Time'].values
            y1 = vib2['Filter_X'].values
            y2 = vib2['Filter_Y'].values
            y3 = vib2['Filter_Z'].values

            dy1 = np.zeros(y1.shape,np.float)
            dy1[0:-1] = np.diff(y1)/np.diff(x)
            dy1[-1] = (y1[-1] - y1[-2])/(x[-1] - x[-2])
            
            dy2 = np.zeros(y2.shape,np.float)
            dy2[0:-1] = np.diff(y2)/np.diff(x)
            dy2[-1] = (y2[-1] - y2[-2])/(x[-1] - x[-2])
            
            dy3 = np.zeros(y3.shape,np.float)
            dy3[0:-1] = np.diff(y3)/np.diff(x)
            dy3[-1] = (y3[-1] - y3[-2])/(x[-1] - x[-2])
            
            dysv = np.sqrt(dy1**2 + dy2**2 + dy3**2)
            
            acel_tmp = [d, max(dy1)/(9.8), max(dy2)/(9.8), max(dy3)/(9.8), 
                        np.sqrt(max(dy1)**2 + max(dy2)**2 + max(dy3)**2)/(9.8)]
            acel.append(acel_tmp)
            
        return pd.DataFrame(acel,columns=['Delay','Acel_X','Acel_Y','Acel_Z','Acel_SV'])
    
def Special_Disp(seed,n,Delay):
    vib, amp = Ideal_Test(seed,n,Delay,absolute=True)
    
    if np.isscalar(Delay) == True:
        peak_out_x, peak_out_y, peak_out_z = Special_ppvFreq(vib, hei_rop=0.1, dis_rop=25)
        
        index_x = peak_out_x[peak_out_x['peak_X'] == max(peak_out_x['peak_X'])].index
        disp_x = peak_out_x['peak_X'][index_x]/(2*math.pi*peak_out_x['Frequency'][index_x])
            
        index_y = peak_out_y[peak_out_y['peak_Y'] == max(peak_out_y['peak_Y'])].index
        disp_y = peak_out_y['peak_Y'][index_y]/(2*math.pi*peak_out_y['Frequency'][index_y])
            
        index_z = peak_out_z[peak_out_z['peak_Z'] == max(peak_out_z['peak_Z'])].index
        disp_z = peak_out_z['peak_Z'][index_z]/(2*math.pi*peak_out_z['Frequency'][index_z])
        
        disp_sv = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)
        
        disp = [Delay, disp_x, disp_y, disp_z, disp_sv]
        
        return pd.DataFrame([disp],columns=['Delay','Disp_X','Disp_Y','Disp_Z','Disp_SV'])
    else:
        disp = []
        
        #vib = vib.reset_index(drop=True)
        for d in Delay:
            vib2 = vib[vib['Delay']==d].iloc[:,0:4].copy()
            vib2 = vib2.reset_index(drop=True)
                     
            peak_out_x, peak_out_y, peak_out_z = Special_ppvFreq(vib2, hei_rop=0.1, dis_rop=25)

            index_x = peak_out_x[peak_out_x['peak_X'] == max(peak_out_x['peak_X'])].index[0]
            disp_x = peak_out_x['peak_X'][index_x]/(2*math.pi*peak_out_x['Frequency'][index_x])

            index_y = peak_out_y[peak_out_y['peak_Y'] == max(peak_out_y['peak_Y'])].index[0]
            disp_y = peak_out_y['peak_Y'][index_y]/(2*math.pi*peak_out_y['Frequency'][index_y])

            index_z = peak_out_z[peak_out_z['peak_Z'] == max(peak_out_z['peak_Z'])].index[0]
            disp_z = peak_out_z['peak_Z'][index_z]/(2*math.pi*peak_out_z['Frequency'][index_z])
            
            disp_sv = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)
            
            disp_tmp = [d, disp_x, disp_y, disp_z, disp_sv]
            disp.append(disp_tmp)
            
            del vib2, peak_out_x, peak_out_y, peak_out_z, disp_x, disp_y, disp_z, disp_sv, disp_tmp
    
        return pd.DataFrame(disp,columns=['Delay','Disp_X','Disp_Y','Disp_Z','Disp_SV'])

def Special_Energy(seed,n,Delay):
    vib, amp = Ideal_Test(seed,n,Delay,absolute=True)
    
    if np.isscalar(Delay) == True:
        peak_out_x, peak_out_y, peak_out_z = Special_ppvFreq(vib, hei_rop=0.1, dis_rop=25)

        eng_x = 0.5*(max(peak_out_x['peak_X'])/1e3)**2
        eng_y = 0.5*(max(peak_out_y['peak_Y'])/1e3)**2
        eng_z = 0.5*(max(peak_out_z['peak_Z'])/1e3)**2
            
        eng_sv = eng_x + eng_y + eng_z
            
        energy_tmp = [d, eng_x, eng_y, eng_z, eng_sv]
        energy.append(energy_tmp)
        eng_org = pd.DataFrame(energy,columns=['Delay','Energy_X','Energy_Y','Energy_Z','Energy_SV'])
        eng = eng_org.copy()
        eng['Energy_X'] = 100*(eng['Energy_X']/max(eng['Energy_X']))
        eng['Energy_Y'] = 100*(eng['Energy_Y']/max(eng['Energy_Y']))
        eng['Energy_Z'] = 100*(eng['Energy_Z']/max(eng['Energy_Z']))
        eng['Energy_SV'] = 100*(eng['Energy_SV']/max(eng['Energy_SV']))
        eng.columns = ['Delay','Energy_%_X','Energy_%_Y','Energy_%_Z','Energy_%_SV']
        
        return eng_org, eng
    else:
        energy = []
        for d in Delay:
            peak_out_x, peak_out_y, peak_out_z = Special_ppvFreq(vib[vib['Delay']==d].iloc[:,0:4], hei_rop=0.1, dis_rop=25)

            eng_x = 0.5*(max(peak_out_x['peak_X'])/1e3)**2
            eng_y = 0.5*(max(peak_out_y['peak_Y'])/1e3)**2
            eng_z = 0.5*(max(peak_out_z['peak_Z'])/1e3)**2
            
            eng_sv = eng_x + eng_y + eng_z
            
            energy_tmp = [d, eng_x, eng_y, eng_z, eng_sv]
            energy.append(energy_tmp)
        eng_org = pd.DataFrame(energy,columns=['Delay','Energy_X','Energy_Y','Energy_Z','Energy_SV'])
        eng = eng_org.copy()
        eng['Energy_X'] = 100*(eng['Energy_X']/max(eng['Energy_X']))
        eng['Energy_Y'] = 100*(eng['Energy_Y']/max(eng['Energy_Y']))
        eng['Energy_Z'] = 100*(eng['Energy_Z']/max(eng['Energy_Z']))
        eng['Energy_SV'] = 100*(eng['Energy_SV']/max(eng['Energy_SV']))
        eng.columns = ['Delay','Energy_%_X','Energy_%_Y','Energy_%_Z','Energy_%_SV']
        
        return eng_org, eng

def Full_Module01(directoryPath, file_in = 'wave_01', file_out = 'analysis_01', n=10, Delay=np.arange(0.000,0.101,0.001)):
    
    seed_f = Load_Seed(directoryPath, file_in,scale=False)
    print('(01/10) - Load SeedWave............. DONE')
    
    #seed_off, offset = Offset(seed_f, offset_lim=0.1)
    #print('(2/9) - Offset... DONE')
    
    # define min max scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # transform data
    s_tmp = seed_f.iloc[:, 1:].copy()
    
    s_tmp = pd.DataFrame(scaler.fit_transform(s_tmp),columns=s_tmp.columns)
    
    #seed_f2 = seed_f.copy()
    #seed_f2.iloc[:,1:] = s_tmp.iloc[:,0:]
    print('(02/10) - Processing Procedure...... DONE')
    
    vib_seed, amp_seed = Ideal_Test(seed_f,n=n,Delay=Delay,absolute=True)
    vib_seed2, amp_seed2 = Ideal_Test(seed_f,n=n,Delay=Delay,absolute=False)
    amp_seed.columns = ['Delay','AbsVal Amp_X','AbsVal Amp_Y','AbsVal Amp_Z','AbsVal Amp_SV']
    print('(03/10) - Ideal Test................ DONE')
                       
    ppv_x = pd.DataFrame()
    ppv_y = pd.DataFrame()
    ppv_z = pd.DataFrame()
    for d in Delay:
        ppvfreq_x, ppvfreq_y, ppvfreq_z = Special_ppvFreq(vib_seed[vib_seed['Delay']==d].iloc[:,0:4])
        ppvfreq_x['Delay'] = d
        ppvfreq_y['Delay'] = d
        ppvfreq_z['Delay'] = d
        
        ppv_x = pd.concat([ppv_x,ppvfreq_x])
        ppv_y = pd.concat([ppv_y,ppvfreq_y])
        ppv_z = pd.concat([ppv_z,ppvfreq_z])
    print('(04/10) - PPV/Frequency............. DONE')
    #print(ppv_y['Delay'].unique())
    
    zc = []
    for dd in Delay:
        indx = np.argmax(ppv_x[ppv_x['Delay'] == dd]['peak_X'])
        fx = ppv_x[ppv_x['Delay'] == dd]['Frequency'].iloc[indx]
        indy = np.argmax(ppv_y[ppv_y['Delay'] == dd]['peak_Y'])
        fy = ppv_y[ppv_y['Delay'] == dd]['Frequency'].iloc[indy]
        indz = np.argmax(ppv_z[ppv_z['Delay'] == dd]['peak_Z'])
        fz = ppv_z[ppv_z['Delay'] == dd]['Frequency'].iloc[indz]
        
        f_tmp = [dd, fx, fy, fz]
        zc.append(f_tmp)
    zc_freq = pd.DataFrame(zc,columns = ['Delay','ZC Freq_X','ZC Freq_Y','ZC Freq_Z'])
    print('(05/10) - Zero Crossing Frequency... DONE')
      
    ### AQUÍ LA ACELERACIÓN 
    acel_g = Special_Acel(seed_f,n,Delay=Delay)
    print('(06/10) - Particle Aceleration...... DONE')
    
    disp_seed = Special_Disp(seed_f,n=n,Delay=Delay)
    print('(07/10) - Particle Displacement..... DONE')
                          
    eng_org, eng_seed = Special_Energy(seed_f,n=n,Delay=Delay)
    print('(08/10) - Energy per Mass Unit...... DONE')
                          
    domfreq_seed = Special_DomFreq(seed_f,n=n,Delay=Delay)
    print('(09/10) - Dominant Frequency........ DONE')   
    
    # create a excel writer object
    filename = os.path.join(directoryPath + file_out + '.xlsx')
    with pd.ExcelWriter(filename) as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        seed_f.to_excel(writer, sheet_name="Seed Wave (mm s)", index=False)
        #seed_off.to_excel(writer, sheet_name="Seed Wave - offset", index=False)
        #offset.to_excel(writer, sheet_name="Offset Values", index=False)
        seed_f.to_excel(writer, sheet_name="Seed Wave - Scaled", index=False)
        vib_seed.to_excel(writer, sheet_name="Seed Wave Superposition (mm s)", index=False)
        
        pd.merge(amp_seed,amp_seed2, how="inner", on=["Delay"]).to_excel(writer, sheet_name="Seed Wave Amplification", index=False)
        
        zc_freq.to_excel(writer, sheet_name="Zero Crossing Frequency (Hz)", startcol=0, index=False)
        ppv_x.to_excel(writer, sheet_name="PPV - Frequency", startcol=0, index=False)
        ppv_y.to_excel(writer, sheet_name="PPV - Frequency", startcol=4, index=False)
        ppv_z.to_excel(writer, sheet_name="PPV - Frequency", startcol=8, index=False)
        domfreq_seed.to_excel(writer, sheet_name="Dominant Frequency (Hz)", index=False)
        acel_g.to_excel(writer, sheet_name="Particle Max Aceleration (mg)", index=False)
        disp_seed.to_excel(writer, sheet_name="Particle Max Displacement (mm)", index=False)
        pd.merge(eng_org, eng_seed, how="inner", on=["Delay"]).to_excel(writer, sheet_name="Energy per Mass Unit (%)", index=False)
        
    print('(10/10) - Excel File Saved.......... DONE') 
    return
