#%% Anomaly Detection
import os
os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

""" Helps in exploring the anamolies using stationary standard deviation
Takes y (pandas.Series): independent variable, window_size (int): rolling window size
and sigma (int): value for standard deviation.
It Returns:a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))"""
def anomalies_rolling_std(x, y, window_size, sigma=1.0):
    # Get MA
    avg = y.rolling(window = window_size, center=False).mean()
    avg.fillna(method='bfill', inplace=True)

    # Convert to list as required
    avg_list = avg.tolist()
    residual = y - avg

    # Calculate the variation in the distribution of the residual
    testing_std = residual.rolling(center=False,window=window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.iloc[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in zip(x,  y, avg_list, rolling_std)
              if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}
# end of anomalies_rolling_std

# df takes first column DateTime, second values and third color
def save_plot(df, window_size, file_name_to_save = None, title =''):
    #
    text_xlabel=df.columns[0]; text_ylabel=df.columns[1]; color_label = df.columns[2]
    df[color_label] = df[color_label].astype('str')

# Calculate few statistics require for plotting
    # Moving average
    y_av = df[text_ylabel].rolling(window = window_size, center=False).mean()
    y_av.fillna(method='bfill', inplace=True)

# plot the Anomaly
    plt.figure(figsize=(15, 8))
    ax =plt.subplot(111)
    # Plot all points  using 'black' color so that non handle points are also plotted, at least
    ax.plot(df.loc[:,text_xlabel], df.loc[:,text_ylabel], color = 'k', marker = '.')
    ax.plot(df.loc[:,text_xlabel], y_av, color='green' , linewidth=6)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)
    # Plot anomaly colors ?plt.plot
    if df[df[color_label] == 'red'].shape[0] > 0:
        ax.plot(df[df[color_label] == 'red'].loc[:,text_xlabel], df[df[color_label] == 'red'].loc[:,text_ylabel], 'r.') # color = 'r', marker = '.'
    if df[df[color_label] == 'blue'].shape[0] > 0:
        ax.plot(df[df[color_label] == 'blue'].loc[:,text_xlabel], df[df[color_label] == 'blue'].loc[:,text_ylabel], 'b.')
    if df[df[color_label] == 'orange'].shape[0] > 0:
        ax.plot(df[df[color_label] == 'orange'].loc[:,text_xlabel], df[df[color_label] == 'orange'].loc[:,text_ylabel], '.', color='orange')

    # add grid and lines and enable the plot
    plt.title(title); plt.grid(True); plt.tight_layout()
    if file_name_to_save is None:
        plt.show()
    else:
        plt.savefig(file_name_to_save)

    plt.close(); plt.gcf().clear()

    del(y_av)
    return
# end of save_plot

# Lets us use the functions. load the data set
data = pd.read_csv('./data/AirPassengers.csv')
data.head()
data.dtypes

#date time conversion
data['TravelDate'] =  pd.to_datetime(data['TravelDate'], format='%m/%d/%Y')
data['Passengers'] = data['Passengers'].astype(float)
data.dtypes # Notice the dtype=’datetime[ns]’

# See the data and get seasonality
data.head(20)

# From above
window_size = 12;
sigma = 2 # Assume as per business

# Get Anomaly events
anomaly_events = anomalies_rolling_std(data['TravelDate'], data['Passengers'], window_size=window_size, sigma= sigma)

# Display the anomaly data
anomaly_events_dt = [dt for dt in  anomaly_events['anomalies_dict']]
anomaly_events_dt[:2]

# plot the results
# preprae data
df = data[['TravelDate', 'Passengers']]
df['COLOR'] = np.tile('black', df.shape[0])
df.loc[df['TravelDate'].isin(anomaly_events_dt),'COLOR'] = 'red'
title = 'Overall plot: Normal in black, Anomaly in red and Moving Average in green color'

# Save the plot
file_name = ''.join(['.Iimages/', 'anomaly_movavg_rolling_std_Passengers.png'])
save_plot(df, window_size, file_name,title)
