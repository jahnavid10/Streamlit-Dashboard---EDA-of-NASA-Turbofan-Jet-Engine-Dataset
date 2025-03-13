import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_preprocess_data(data):
    df = pd.read_csv(data, sep=' ', header=None).dropna(axis=1, how='all')
    column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 
                    'operational_setting_2', 'operational_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df.columns = column_names
    return df

def calculate_rul(df):
    EOL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    EOL.rename(columns={'time_in_cycles': 'EOL'}, inplace=True)
    df = df.merge(EOL, on='unit_number')
    df['RUL'] = df['EOL'] - df['time_in_cycles']
    return df

def plot_histograms(df):
    plt.figure(figsize=(15,20))
    for i in np.arange(1, len(df.columns)):
        ax = plt.subplot(5, 6, i)
        ax.hist(df.iloc[:, i], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(df.columns[i], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    st.pyplot(plt)

def plot_correlation_matrix(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(58,58))
    sns.set(font_scale=4, font="Times New Roman")
    g = sns.heatmap(df[top_corr_features].corr(), cmap="RdYlGn", linewidths=0.1, annot=True, annot_kws={"size":35})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=35)
    g.set_yticklabels(g.get_xmajorticklabels(), fontsize=35)
    st.pyplot(plt)

st.title('Engine Data Analysis')
dataset_options = ['Dataset FD001', 'Dataset FD002', 'Dataset FD003', 'Dataset FD004']
selected_dataset = st.selectbox('Select a Dataset', dataset_options)

uploaded_file = st.file_uploader("Upload data file", key="uploader")
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    st.write(df.head())
    if st.button('Calculate RUL'):
        df = calculate_rul(df)
        st.write(df.head())
    if st.button('Show Histograms'):
        plot_histograms(df)
    if st.button('Show Correlation Matrix'):
        plot_correlation_matrix(df)
