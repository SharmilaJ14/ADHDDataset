import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import streamlit as st

Control=[]
ADHD=[]
def importDataset():
    column=['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2']
    os.chdir('/Users/sharmilajayasankaran/Control')
    flist1=[f for f in os.listdir()]
    for i in flist1:
        df=pd.read_csv(i)
        df.drop(df.iloc[:,0:1],inplace=True,axis=1)
        Control.append(df)
    os.chdir('/Users/sharmilajayasankaran/ADHD')
    flist2=[f for f in os.listdir()]
    for i in flist2:
        df=pd.read_csv(i)
        df.drop(df.iloc[:,0:1],inplace=True,axis=1)
        ADHD.append(df)
    for i in range(len(ADHD)): 
        ADHD[i].columns=column
    for i in range(len(Control)):
        Control[i].columns=column
    return flist1,flist2

def Normalize():
    length1=[]
    length2=[]
    for i in range(0,len(ADHD)):
        length1.append(len(ADHD[i]))
    for i in range(0,len(Control)):
        length2.append(len(Control[i]))
    minLength=min(min(length1),min(length2))
    for i in range(0,len(ADHD)):
        ADHD[i]=ADHD[i].head(minLength)
    for i in range(0,len(Control)):
        Control[i]=Control[i].head(minLength)
    return minLength

def calculate_dist_corr(v1,v2):
     dist_corr = scipy.spatial.distance.correlation(v1,v2)
     return dist_corr

def correlationADHD():
    cor1=[]
    for i in range(len(ADHD)):
        spearman_corr = ADHD[i].corr(method='spearman')
        kendall_corr = ADHD[i].corr(method='kendall')
        cor1.append(spearman_corr)
    return cor1

def correlationControl():
    cor1=[]
    for i in range(len(Control)):
        spearman_corr = Control[i].corr(method='spearman')
        kendall_corr = Control[i].corr(method='kendall')
        cor1.append(spearman_corr) 
    return cor1

def topomap(array):
    column=['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2']
    eeg_data = array[0].to_numpy()
    sfreq = 128.0  
    info = mne.create_info(column, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg_data.T, info)
    time_point = 0.5
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(raw.get_data()[:, int(time_point * sfreq)], pos=raw.info, names=column, axes=ax, show=False)
    plt.title(f'Topomap for Control at {time_point} seconds')
    return fig

def MedianMatrix(array):
    medianA=[]
    for i in range(0,len(array)):
        median=[]
        for col in array[i].columns:
            median.append(np.median(array[i][col]))
        medianA.append(median)
    return medianA

def SpectralClusteringD(correlation1):
    all_labels = []
    for subject_corr_matrix in correlation1:
        similarity_matrix = rbf_kernel(subject_corr_matrix, gamma=1.0)
        n_clusters = 5
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = spectral_clustering.fit_predict(similarity_matrix)
        all_labels.append(labels)
    for subject_index, labels in enumerate(all_labels):
        plt.scatter(range(len(labels)), labels, cmap='viridis')
        plt.title(f'Cluster Results for Subject {subject_index + 1}')
        plt.xlabel('Data Point Index')
        plt.ylabel('Cluster Label')
        plt.show()
    return all_labels
    

st.header("ML Lab Test 2")
st.subheader("Q1)ADHD dataset")
st.text("""In this webapp, we present the code,inference and analysis of the ADHD Dataset. 
        Presented by Harini K R(21PD09) and Sharmila J(21PD33)""")
st.subheader("Notebook")
st.text("Importing the necessary modules")
st.code("""import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel""")
st.text("Importing the dataset")
st.code("""column=['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2']
os.chdir('/Users/sharmilajayasankaran/Control')
flist1=[f for f in os.listdir()]
for i in flist1:
    df=pd.read_csv(i)
    df.drop(df.iloc[:,0:1],inplace=True,axis=1)
    Control.append(df)
os.chdir('/Users/sharmilajayasankaran/ADHD')
flist2=[f for f in os.listdir()]
for i in flist2:
    df=pd.read_csv(i)
    df.drop(df.iloc[:,0:1],inplace=True,axis=1)
    ADHD.append(df)
for i in range(len(ADHD)): 
    ADHD[i].columns=column
for i in range(len(Control)):
    Control[i].columns=column""")
importDataset()
st.text("A sample Dataframe from ADHD data")
st.dataframe(ADHD[0])
st.text("A Sample dataframe from Control dataset")
st.dataframe(Control[0])
st.text("Equalizing the number of rows in all datasets")
st.code(""" length1=[]
length2=[]
for i in range(0,len(ADHD)):
    length1.append(len(ADHD[i]))
for i in range(0,len(Control)):
    length2.append(len(Control[i]))
minLength=min(min(length1),min(length2))
for i in range(0,len(ADHD)):
    ADHD[i]=ADHD[i].head(minLength)
for i in range(0,len(Control)):
    Control[i]=Control[i].head(minLength)""")
Normalize()
st.text("A sample Dataframe from ADHD data after row equalization")
st.dataframe(ADHD[0])
st.text("A Sample dataframe from Control dataset after row equalization")
st.dataframe(Control[0]) 
st.subheader("Some Visualisations")
st.subheader("Correlation matrix")
st.text("Non-linear correlation matrix has been found using Spearman method")
st.code(""" cor1=[]
for i in range(len(ADHD)):
    spearman_corr = ADHD[i].corr(method='spearman')
    cor1.append(spearman_corr)""")
corrADHD=correlationADHD()
st.table(corrADHD[0])
fig=sns.heatmap(ADHD[0])
#st.pyplot(fig)
st.text("Inference:")
st.code(""" cor2=[]
for i in range(len(Control)):
    spearman_corr = Control[i].corr(method='spearman')
    cor2.append(spearman_corr)""")
corrC=correlationControl()
st.table(corrC[0])
fig=sns.heatmap(corrC[0])
#st.pyplot(fig)
st.text("Inference: ")
st.subheader("Topomap plotting and analysis")
st.code("""column=['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2']
eeg_data = array[0].to_numpy()
sfreq = 128.0  
info = mne.create_info(column, sfreq=sfreq, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
raw = mne.io.RawArray(eeg_data.T, info)
time_point = 0.5
fig, ax = plt.subplots()
mne.viz.plot_topomap(raw.get_data()[:, int(time_point * sfreq)], pos=raw.info, names=column, axes=ax, show=False)
plt.title(f'Topomap for Control at {time_point} seconds')
plt.show() """)
st.text("Topomap of ADHD Dataset")
st.pyplot(topomap(ADHD))
st.text("Topomap of Control Dataset")
st.pyplot(topomap(Control))











