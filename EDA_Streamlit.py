3# Importing libraries
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import matplotlib.ticker as ticker

# Dictionaries for file paths
data_paths = {
    'train': {
        1: r"C:/Users/JAHNAVI/Desktop/archive/CMaps/train_FD001.txt",
        2: r"C:/Users/JAHNAVI/Desktop/archive/CMaps/train_FD002.txt",
        3: r"C:/Users/JAHNAVI/Desktop/archive/CMaps/train_FD003.txt",
        4: r"C:/Users/JAHNAVI/Desktop/archive/CMaps/train_FD004.txt"
    },
    'test': {
        1: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\test_FD001.txt",
        2: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\test_FD002.txt",
        3: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\test_FD003.txt",
        4: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\test_FD004.txt"
    },
    'rul': {
        1: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\RUL_FD001.txt",
        2: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\RUL_FD002.txt",
        3: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\RUL_FD003.txt",
        4: r"C:\Users\JAHNAVI\Desktop\archive\CMaps\RUL_FD004.txt"
    }
}

# Define the common column names for train and test datasets
columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'S_{i}' for i in range(1, 22)]

# Function to load data based on dataset number
@st.cache_data
def load_data(dataset_number):
    train_path = data_paths['train'][dataset_number]
    test_path = data_paths['test'][dataset_number]
    rul_path = data_paths['rul'][dataset_number]

    train_data = pd.read_csv(train_path, sep=' ', header=None).dropna(axis=1, how='all')
    train_data.columns = columns
    test_data = pd.read_csv(test_path, sep=' ', header=None).dropna(axis=1, how='all')
    test_data.columns = columns
    rul_data = pd.read_csv(rul_path, sep=' ', header=None).dropna(axis=1, how='all')
    rul_data.columns = ['RUL']

    # Apply any common preprocessing if necessary, like setting column names
    return train_data, test_data, rul_data

# Streamlit app
st.title('NASA Turbofan Jet Engine Data')
st.write(""" Data sets consists of multiple multivariate time series. 
Each data set is further divided into training and test subsets. Each time series is from a different engine. 
Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, 
i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. 
These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. 
In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time 
prior to system failure.
Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data consists of 26 columns of numbers, separated by spaces. 
Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  21

""")

st.write("# Exploratory Data Analysis")
st.write("## Dataset Selector")

# Allow user to select the dataset number
dataset_number = st.selectbox("Select Dataset Number", [1, 2, 3, 4])

# Display the selected dataset number
st.write(f"Selected Dataset: {dataset_number}")

# Load data based on selected dataset number
train, test, rul = load_data(dataset_number)

# Display dataset-specific information
if dataset_number == 1:
    st.write("""Dataset 1 consists of:
\n - Train trjectories: 100
\n - Test trajectories: 100
\n - Conditions: ONE (Sea Level)
\n - Fault Modes: ONE (HPC Degradation)""")
elif dataset_number == 2:
    st.write("""Dataset 2 consists of:
\n - Train trjectories: 260
\n - Test trajectories: 259
\n - Conditions: SIX
\n - Fault Modes: ONE (HPC Degradation)""")
elif dataset_number == 3:
    st.write("""Dataset 3 consists of:
\n - Train trjectories: 100
\n - Test trajectories: 100
\n - Conditions: ONE (Sea Level) 
\n - Fault Modes: TWO (HPC Degradation, Fan Degradation)""")
elif dataset_number == 4:
    st.write("""Dataset 4 consists of:
\n - Train trjectories: 248
\n - Test trajectories: 249
\n - Conditions: SIX
\n - Fault Modes: TWO (HPC Degradation, Fan Degradation)""")

# Show loaded data
st.write("""### Train Data:
\n The Train data that is generated here has NaN values removed and column names assigned.""")
st.write(train)
st.write("### Shape of the Train Data:")
st.write(train.shape)
st.write("### Info about the datatypes of Train Data:")
st.write("This gives the information of the datatypes of each column.")
st.write(train.dtypes)
st.write("#### Unique values of Train Data:")
st.write(train.nunique())
# Allow user to select the variant of describe
st.write("""### This option enables to generate Descriptive Statistics of the required columns.
\n #### Here's what each statistic represents:
\n - count: Number of non-null observations.
\n - mean: Mean of the values.
\n - std: Standard deviation of the values.
\n - min: Minimum value.
\n - 25%: First quartile (25th percentile).
\n - 50%: Median (50th percentile).
\n - 75%: Third quartile (75th percentile).
\n - max: Maximum value.""")
variant = st.selectbox("Select Describe Variant", ["Full", "Unit Number & Time in Cycles", "Settings", "Sensor Data"])

# Display the selected variant of describe
if variant == "Full":
    st.write("Full Describe:")
    st.write(train.describe().transpose())
elif variant == "Unit Number & Time in Cycles":
    st.write("Describe for Unit Number & Time in Cycles:")
    st.write(train.loc[:,['unit_number','time_in_cycles']].describe().transpose())
elif variant == "Settings":
    st.write("Describe for Settings:")
    st.write(train.loc[:,['setting_1','setting_2','setting_3']].describe().transpose())
elif variant == "Sensor Data":
    st.write("Describe for Sensor Data:")
    st.write(train.loc[:,'S_1':].describe().transpose())

st.write("""### Calculation of Maximum Time Cycles:
\n It records the highest number of cycles it has been through before failure or maintenance for each engine. """)

# Calculation of Maximum time cycles for each engine(unit number)
@st.cache_data
def calculate_max_time_cycles(df):
    # Group the data by 'unit_number' and calculate the maximum for the specified columns
    max_time_cycles = df.groupby(['unit_number'])[["unit_number","time_in_cycles"]].max()

    return max_time_cycles

# Calculate maximum time cycles
max_time_cycles = calculate_max_time_cycles(train)

# Display the result
st.write("Maximum Time Cycles for Each Engine (Unit Number):")
st.write(max_time_cycles)

# Define a function to plot turbofan engines lifetime
@st.cache_data
def plot_engines_lifetime(df):
    plt.figure(figsize=(20,60))
    ax = df['time_in_cycles'].plot(kind='barh', width=0.8, stacked=True, align='center')
    plt.title('Turbofan Engines Lifetime', fontweight='bold', size=30)
    plt.xlabel('Time in cycles', fontweight='bold', size=20)
    plt.xticks(size=15)
    plt.ylabel('Unit number', fontweight='bold', size=20)
    plt.yticks(size=15)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# Display plot button

if st.checkbox('Plot Turbofan Engines Lifetime'):
    st.write("""The plot titled "Turbofan Engines Lifetime" visually represents the operational lifetimes of turbofan engines, 
with each bar on the horizontal chart indicating the number of cycles each engine operated before being taken out of service or failure. 
This chart helps in analyzing engine performance, identifying variations and outliers in engine lifetimes, and aids in lifecycle management 
by providing insights into when engines might typically require maintenance or replacement. Such data is crucial for predictive maintenance 
and operational planning in industries that rely heavily on engine performance and reliability. """)
    plot_engines_lifetime(max_time_cycles)

# Define a function to plot distribution
@st.cache_data
def plot_distribution(data, column, bins=20, height=6, aspect=2):
    # Create the distribution plot with KDE
    sns.displot(data[column], kde=True, bins=bins, height=height, aspect=aspect)
    plt.xlabel('Max Time Cycle')

    # Format x-axis ticks to display as integers
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    st.pyplot(plt)

# Checkbox for distribution plot option
if st.checkbox('Distribution Plot'):
    st.write("""This histogram with a fitted curve illustrates the distribution of the maximum operational cycles (time in cycles) 
for a set of turbofan engines. The x-axis represents the maximum time in cycles an engine has operated, and the y-axis shows the 
count of engines that reached those cycles. The peak of the Histogram represents the range of time cycles within which the most engines have.""")
    # Plot the distribution
    plot_distribution(max_time_cycles, 'time_in_cycles', bins=20, height=6, aspect=2)

st.write("""### Calculation fo RUL
\n RUL values for each time cycle of every unit number is calculated.""")

# Calculating RUL values
@st.cache_data
def calculate_RUL(df):
    train_grouped_by_unit = df.groupby(by='unit_number')
    max_time_cycles = train_grouped_by_unit['time_in_cycles'].max()
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_in_cycles']
    merged = merged.drop("max_time_cycle", axis=1)
    return merged

# Calculate RUL values
train_rul = calculate_RUL(train)

# Display the resulting DataFrame
st.write("Train Data with RUL:")
st.write(train_rul)

# Checkbox to trigger describe
if st.checkbox("### Descriptive Statisctics of RUL"):
    # Describe the DataFrame
    st.write("Description of RUL column:")
    st.write(train_rul.loc[:,['RUL']].describe().transpose())

st.write("### Plots of Train data")
st.write("Different types of plots and Heat maps are generated for the Train data with RUL based on which findings are generated.")

# Plotting Train data

# Define a function to plot sensor histograms
@st.cache_data
def plot_sensor_histograms(df):
    sns.set()
    fig = plt.figure(figsize=[15, 10])
    cols = df.columns
    cnt = 1
    for col in cols:
        plt.subplot(8, 4, cnt)
        sns.histplot(df[col], color='blue')
        cnt += 1
    plt.tight_layout()
    st.pyplot(plt)

# Define a function to plot heatmap
@st.cache_data
def plot_heatmap(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Create a heatmap and annotate it
    plt.figure(figsize=(20, 20))  # Set the size of the figure directly
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f", square=True, cbar_kws={'shrink': .8})

    # Adding labels (assuming column names are appropriate to be used directly)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x labels for better readability
    plt.yticks(fontsize=10) 

    plt.title('Correlation Matrix')  
    st.pyplot(plt)


# Define a function to plot boxplots
@st.cache_data
def plot_boxplots(df):
    # Number of features (i.e., columns)
    num_features = len(df.columns)

    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 2))

    # If only one feature, turn axes into a list
    if num_features == 1:
        axes = [axes]

    # Iterate through features and create box plots
    for ax, feature in zip(axes, df.columns):
        sns.boxplot(data=df, x=feature, ax=ax, color='skyblue')
        ax.set_title(f'Box plot of {feature}')
        ax.set_xlabel('')
        ax.set_ylabel(feature)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(plt)

# Checkboxes to toggle visibility of plots
show_histograms = st.checkbox("Show Sensor Histograms")
show_heatmap = st.checkbox("Show Correlation Heatmap")
show_boxplots = st.checkbox("Show Boxplots")

# Display the plots based on the checkbox selections
if show_histograms:
    st.write("""### Histograms for all columns of Train data
\n Histograms provide insights into the central tendency, dispersion, and shape of a dataset. 
They are used to identify patterns, detect outliers, assess data quality, and make informed 
decisions regarding preprocessing and feature engineering. They serve as a starting point for understanding the underlying structure 
of data, aiding in the selection of relevant features, comparison of distributions between groups, and identification of potential data issues.""")
    plot_sensor_histograms(train_rul)

if show_heatmap:
    st.write("""### Correlation Heatmap for the Train Data
\n Correlation heatmaps are essential tools in data analysis for visually representing the correlation between variables in a dataset. 
By displaying correlation coefficients as a color-coded matrix, heatmaps provide quick and intuitive insights into the relationships between variables. 
These visualizations enable helps to identify patterns, dependencies, and multicollinearity within the data.""")
    plot_heatmap(train_rul)

if show_boxplots:
    st.write("""### Boxplots for all columns of Train data
\n Boxplots are indispensable tools in data analysis for visually summarizing the distribution of numerical data and identifying potential outliers. 
They provide a concise representation of key statistical measures such as the median, quartiles, and range of a dataset. 
Boxplots offer insights into the central tendency, spread, and variability of the data across different groups or categories, 
making them invaluable for comparative analysis. By displaying the distribution of data as quartiles and visualizing outliers as
 individual data points beyond the whiskers, boxplots facilitate the detection of anomalous observations. """)
    plot_boxplots(train_rul)

st.write("""
### Here are a few key points:
- From different Histograms, it maybe observed that few features have constant values throughout which can also be observed in descriptive statistics. 
         Such features donot contribute towards predictive performance of the model. Hence, such features can be dropped from the dataframe.
- The correlation matrix reveals significant correlations between several variables with RUL (Remaining Useful Life).
         From Correlation Heatmap it can be observed that few feartures have almost no correlation with RUL, 
         indicating that these might not be useful in predicting remaining life.
- The box plots highlight the presence of outliers and the spread of data. Few features have noticeable outliers,
          which suggests extreme values that may need further investigation or treatment during data preprocessing. 
""")

st.write("""### Dropping of non-relevant columns
\n Based on above keypoints:
- features with constant values throughout are dropped.
- An option to select correlation threshold is provided to drop certain features based on their correlation coefficient.
- time_in_cycles column is also dropped""")
# Function to drop unnecessary columns
@st.cache_data
def drop_unnecessary_columns(df, correlation_threshold=0):
    corr = df.corr()  # Correlation matrix
    abs_corr = corr.abs()  # Absolute value of correlation matrix
    cols_to_drop_by_sum = abs_corr.columns[abs_corr.sum() <= 1]  # Columns weakly correlated across all variables
    corr_with_rul = abs_corr['RUL']  # Specific focus on correlation with 'RUL'
    cols_to_drop_by_rul = corr_with_rul[corr_with_rul < correlation_threshold].index.tolist()  # Threshold check
    cols_to_drop = list(set(cols_to_drop_by_sum.tolist() + cols_to_drop_by_rul))  # Combine and remove duplicates
    if 'RUL' in cols_to_drop:
        cols_to_drop.remove('RUL')  # Ensuring RUL is not dropped
    return df.drop(columns=cols_to_drop), cols_to_drop

# User input for correlation threshold
threshold = st.slider("Select Correlation Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("""Setting a correlation threshold helps in determining which features are significantly related, 
aiding in feature selection by eliminating redundant variables, thereby reducing multicollinearity and improving 
model stability. This process enhances model interpretability and computational efficiency, and also reduces the 
risk of overfitting, making models more generalizable. However, the choice of the threshold is crucial; too high 
may cause loss of important features, while too low could retain unnecessary noise, thus requiring careful balance 
and domain-specific adjustments. """)

# Processing
train_rul_cleaned, dropped_columns = drop_unnecessary_columns(train_rul, correlation_threshold=threshold)

# Display results
st.write("### Cleaned Data", train_rul_cleaned)
st.write("### Dropped Columns", dropped_columns)

# New cleaned Heatmap
st.write("### New Heatmap of Cleaned Data")
# Function to plot a cleaned heatmap
@st.cache_data
def plot_cleaned_heatmap(df):
    corr = df.corr()  # Correlation matrix
    plt.figure(figsize=(15, 15))  # Adjusting figure size for better fit in Streamlit
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f", square=True, cbar_kws={'shrink': .8})
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x labels for readability
    plt.yticks(fontsize=10)  # Adjust y labels fontsize
    plt.title('Correlation Heatmap of Cleaned Data')
    st.pyplot(plt) 

# Button to generate heatmap
if st.button('Generate Heatmap'):
    plot_cleaned_heatmap(train_rul_cleaned)

# DATA PREPROCESSING
st.write("# DATA PREPROCESSING")
st.write("""Data preprocessing is a step in the data mining and data analysis process that takes raw data and transforms it into a format 
         that can be understood and analyzed by computers and machine learning.                                                                                                                                                            
Let us Analyze the Test and rul data and also the train data after EDA is performed""")

st.write("### Making Train and Test data and target variables (x_train, x_test, y_train and y_test)")
st.write("Obtaining x_train and y_train data:")

# Function to create x_train and y_train
@st.cache_data
def create_x_y_train(df_cleaned):
    x_train = df_cleaned.drop(columns=['RUL'])
    y_train = df_cleaned[['RUL']].copy()
    return x_train, y_train

x_train, y_train = create_x_y_train(train_rul_cleaned)
st.session_state.x_train = x_train
st.session_state.y_train = y_train

# Display x_train and y_train
st.write("### x_train", st.session_state.x_train)
st.write("### y_train", st.session_state.y_train)

# Options to view shape and describe of x_train and y_train
if st.checkbox('Show Shape and Description of x_train'):
    st.write("Shape of x_train:", st.session_state.x_train.shape)
    st.write("Description of x_train:", st.session_state.x_train.describe().transpose())

if st.checkbox('Show Shape and Description of y_train'):
    st.write("Shape of y_train:", st.session_state.y_train.shape)
    st.write("Description of y_train:", st.session_state.y_train.describe().transpose())

# Making of x_test and y_test data
@st.cache_data
def merge_rul_into_test(test_data, rul_df, x_train):
    # Assume each RUL value in RUL_df corresponds to the respective unit by their indices
    rul_df['unit_number'] = rul_df.index + 1  # Create a 'unit_number' column if it's aligned by index
    # Merge RUL values into the test data based on 'unit_number'
    merged_data = test_data.merge(rul_df, on='unit_number', how='left')
    # Calculate the maximum cycle number per unit to determine the last operational cycle
    merged_data['max_cycle'] = merged_data.groupby('unit_number')['time_in_cycles'].transform(max)
    # Calculate the RUL for each cycle as the difference between the last cycle and current cycle plus RUL at last cycle
    merged_data['RUL'] = (merged_data['max_cycle'] - merged_data['time_in_cycles']) + merged_data['RUL']
    # Cleanup by removing the temporary 'max_cycle' column
    merged_data.drop(columns=['max_cycle'], inplace=True)
    
    # Add any missing columns to merged_data that are present in x_train but not in merged_data
    missing_cols = set(x_train.columns) - set(merged_data.columns)
    for col in missing_cols:
        merged_data[col] = 0  # Or use np.nan or an appropriate placeholder value if missing value imputation is needed
    
    # Reorder and select only the columns as in x_train
    x_test = merged_data.reindex(columns=x_train.columns)

    # Separate into RUL DataFrame and remaining columns DataFrame
    rul_values_df = merged_data[['RUL']].copy()
    remaining_columns_df = merged_data.drop(columns=['RUL'])

    return rul_values_df, x_test


y_test, x_test = merge_rul_into_test(test, rul, x_train)

# Update session state

st.session_state.x_test = x_test
st.session_state.y_test = y_test
st.session_state.data_loaded = True


st.write("### x_test")
st.dataframe(x_test)

st.write("### y_test")
st.dataframe(y_test)

# Options to view shape and describe of x_test and y_test
if st.checkbox('Show Shape and Description of x_test'):
    st.write("Shape of x_test:", st.session_state.x_test.shape)
    st.write("Description of x_test:", st.session_state.x_test.describe().transpose())

if st.checkbox('Show Shape and Description of y_test'):
    st.write("Shape of y_test:", st.session_state.y_test.shape)
    st.write("Description of y_test:", st.session_state.y_test.describe().transpose())


# Function to scale data
@st.cache_data
def scale_data(x_train, x_test, method='standard'):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler()
    }

    if method not in scalers:
        raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', 'maxabs', or 'robust'.")

    scaler = scalers[method]
    scaler.fit(x_train)  # Fit on training data only

    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    return x_train_scaled, x_test_scaled

# Function to plot histograms for sensor data
@st.cache_data
def plot_sensor_histograms(df, title):
    sns.set(style="whitegrid")
    st.subheader(title)
    fig, axes = plt.subplots(nrows=len(df.columns) // 4 + 1, ncols=4, figsize=(20, 4 * (len(df.columns) // 4 + 1)))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[idx], color='blue', kde=True)
        axes[idx].set_title(col)

    for ax in axes[idx+1:]:
        ax.set_visible(False)

    st.pyplot(fig)

st.title("Data Scaling and Visualization Tool")
st.write("""
### Scaling of x_train and x_test
Scaling after EDA is a deliberate practice that ensures the preprocessing aligns with the specific characteristics and needs of the dataset,
as revealed through thorough exploratory analysis. This step is critical to prepare the data adequately for subsequent modeling and to enhance
the overall effectiveness and fairness of the predictive models.
""")

method = st.selectbox(
    "Select a scaling method:",
    ['Choose an option', 'standard', 'minmax', 'maxabs', 'robust'],
    index=0  # default to placeholder 'Choose an option'
)

if method != 'Choose an option':
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test, method=method)
    
    if st.checkbox('Show Scaled Training Data and Histograms'):
        st.write("Scaled Training Data:")
        st.dataframe(x_train_scaled)
        plot_sensor_histograms(x_train_scaled, "Training Data - Scaled Histograms")

    if st.checkbox('Show Scaled Testing Data and Histograms'):
        st.write("Scaled Testing Data:")
        st.dataframe(x_test_scaled)
        plot_sensor_histograms(x_test_scaled, "Testing Data - Scaled Histograms")
else:
    st.info("Please choose a scaling method to proceed.")