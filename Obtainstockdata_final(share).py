# Libraires
import numpy as np
import pandas as pd
import yfinance as yf

# Analysis Functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA # Used for PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import seaborn as sns



def main_function(df):
    
    main_df = pd.DataFrame()
    
    start_date = df['transaction_date']
    end_date = df['plus_forty_date']
    ticker = df['ticker']
    congress_name = df['combined_houses']
    owner = df['owner']
    trans_type = df['type']
    amount = df['amount']
    
    
    # loop through the data frame and obtain the financial data.  
    for (start, end, tick, congress, owner, trans, amnt) in zip(start_date, end_date, ticker, 
                                                                     congress_name, owner, trans_type, amount):
    
        single_transaction, missing_data = get_transactions(start, end, tick, congress, owner, trans, amnt)
        ten_day= ten_days_increase(single_transaction)
        twenty_day = twenty_days_increase(single_transaction)
        thirty_day = thirty_days_increase(single_transaction)
        
        # Concatenate the dataframes 
        main_df = main_df.append(thirty_day)
        
    return main_df


        
def add_days(combined_df):
    """Creates a column for 40 days later"""

    combined_df['plus_forty_date'] = combined_df['transaction_date'] + pd.Timedelta(days=50)

    return combined_df


def combine_representatives(combined_df):
    """Creates a column that combines the senator and representative columns and drops their columns"""

    combined_df['combined_houses'] = combined_df[['senator', 'representative']].bfill(axis=1).iloc[:, 0]
    combined_df = combined_df.drop(columns = ['senator', 'representative'], axis = 1)

    return combined_df


def get_transactions(start, end, tick, congress, owner, trans, amnt):
    """Obtain the transactions for a given number of days""" 

    no_data = [] 
    temp_dict = {}
    
           
    stock_info = yf.Ticker(tick)
    hist_data = stock_info.history(start = start, end = end, interval = '1d')

    # Add to dictionary if financial data is present.
    if hist_data.empty != True:

        temp_dict['name'] = congress
        temp_dict['owner'] = owner
        temp_dict['transaction'] = trans
        temp_dict['amount_invest'] = amnt
        temp_dict['ticker'] = tick
        # Obtain Close values from dataframe
        close_history = hist_data['Close']
        temp_dict['tick'] = close_history
                
    # Collect name if there is no information
    else:
        no_data.append(congress)

    return temp_dict, no_data

      
def ten_days_increase(get_stock_data):
    """Obtain the date 10 training days after"""
    
    if get_stock_data != {}:
        
        new_dict = {}
        date_value = get_stock_data['tick']
        
        if len(date_value) > 10:
        
            # initial day and 10th day
            initial_value = date_value[0]
            ten_day_value = date_value[9]

            # Obtain the initial date
            initial_date = date_value.index[0]
            get_stock_data['Date'] = initial_date

            # 10 day price change
            price_change = ((float(ten_day_value)-float(initial_value))/initial_value)*100

            # Add the price change
            get_stock_data['10_day_change'] = price_change

            return get_stock_data
        

def twenty_days_increase(get_stock_data2):
    """Obtain the date 20 training days after"""
    
    if get_stock_data2 != {}:
        
        new_dict = {}
        date_value = get_stock_data2['tick']
        
        if len(date_value) > 20:
        
            # initial day and 20th day
            initial_value = date_value[0]
            twenty_day_value = date_value[19]

            # 20 day price change
            price_change = ((float(twenty_day_value)-float(initial_value))/initial_value)*100

            # Add the price change
            get_stock_data2['20_day_change'] = price_change

            return get_stock_data2


def thirty_days_increase(get_stock_data3):
    """Obtain the date 30 training days after"""

    if get_stock_data3 != {}:
        
        new_dict = {}
        date_value = get_stock_data3['tick']
        
        if len(date_value) > 30:
        
            # initial day and 30th day
            initial_value = date_value[0]
            thirty_day_value = date_value[29]

            # 30 day price change
            price_change = ((float(thirty_day_value)-float(initial_value))/initial_value)*100

            # Drop the tick column
            get_stock_data3.pop('tick')

            # Add the price change
            get_stock_data3['30_day_change'] = price_change

            # Create a dataframe from the dictionary
            single_trans_dict = pd.DataFrame.from_dict([get_stock_data3])

            return single_trans_dict
           
            
#-------------------------------------------------------------------------------------------------

# Reference: https://towardsdatascience.com/rfmt-segmentation-using-k-means-clustering-76bc5040ead5
def determine_knee(scaled_features):
    """Returns the knee method to determine the number of clusters"""
    
    sse = {}
    
    for k in range(1, 11):
        km = KMeans(
            n_clusters = k, random_state=1
            )
        km.fit(scaled_features)
        sse[k] = km.inertia_

        # plot
    plt.title('The Elbow Method')
    plt.xlabel('k'); plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()
        

    
    
def calc_silhouette(clusters, km, transformed_values):
    """Calculate the silhouette score"""
    
    # Fit the model
    km.fit_predict(transformed_values)
    
    # Calculate the Silhouette score
    score = silhouette_score(transformed_values, km.labels_, metric = 'euclidean')
    
    print(f"The silhouette score for {clusters} clusters is: {score}.")
    
    return


# Reference: https://towardsdatascience.com/rfmt-segmentation-using-k-means-clustering-76bc5040ead5
def clustering(n_clusters, transformed_values, columns):
    """Performs K means clustering"""
    
    # Create dataframe
    X_clust = pd.DataFrame(transformed_values, columns=columns)
    
    km = KMeans(n_clusters = n_clusters, 
               random_state =123, 
               n_init = 500, 
               max_iter = 300)
    km.fit(transformed_values)
    
    # Assign cluster labels
    cluster_labels = km.labels_
    
    # Assign cluster labels to the original pre_transformed data set
    data_clusters = X_clust.assign(Cluster = cluster_labels)
    
    # Group Dataset by k-means cluster
    data_clusters = data_clusters.groupby(['Cluster']).agg('mean')
    
    # Determine the cluster size
    cluster_size = X_clust.assign(Cluster = cluster_labels).groupby(cluster_labels)['owner_dependent'].count()
    
    # Table of clusters
    cluster_table = data_clusters.iloc[:,:21]
    
    print(f"The cluster sizes are:") 
    print(cluster_size)
    print("\n")
    print(f"The clusters are:")
    print(cluster_table)
    print("\n")
        
    # Cluster heatmap
    # Initialize a plot with a figure size of 8 by 2 inches 
    plt.figure(figsize=(20, 10))
    # Add the plot title
    plt.title('Relative importance of attributes')
    # Plot the heatmap
    sns.heatmap(data=data_clusters.iloc[:,:59], fmt='.2f', cmap='RdYlGn', xticklabels=True)
    plt.show()
    
    return km
    
    
# Implement local outlier factor https://medium.com/learningdatascience/anomaly-detection-techniques-in-python 50f650c75aaf#:~:text=Anomaly%20Detection%20Techniques%20in%20Python%201%20Local%20Outlier,dataset.%20...%203%20One-Class%20Support%20Vector%20Machines.%20
