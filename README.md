# ProSTformer_code
# Datasets
download nycbike datasets to folder data_nycbike
 from https://pan.baidu.com/s/1i4N3xuqFrppRgg0Gd6yHUQ  password: 1234

download nyctaxi datasets to folder data_nyctaxi from https://pan.baidu.com/s/1dW3UyYJKL-YLgw7a6ujuLw  password: 1234

## data_x_30, data_x_60, data_x_90
video format shape: (batch, historical_records, channel, height, width) # channel 0: outflow  channel 1: inflow

graph format shape: (batch, historical_records, channel, nodes) # nodes= height*width

the last four weeks are test dataset, i.e., test_dataset: data_x_30[-48\*4\*7:,], data_x_60[-24\*4\*7:,], data_x_90[-16\*4\*7:,], where 48/24/16 means one day (24h) is partitioned according to 30 min, 60 min, and 90 min.

## data_y_30, data_y_60, data_y_90
shape: (batch, channel, height, width) 

## edge matrix

adj_max.npy shape: (nodes, nodes) 


## ext_features 
shape: (batch, features); columns: [temperature, wind, weekend, one-hot weather condition, one-hot weekday], 

where one-hot weather condition includes[  Cloudy	Cloudy / Windy	Drizzle and Fog	Fair	Fair / Windy	Fog	Haze	Heavy Rain	Heavy Rain / Windy	Heavy Snow	Heavy T-Storm	Light Drizzle	Light Drizzle / Windy	Light Freezing Drizzle	Light Freezing Rain	Light Rain	Light Rain / Windy	Light Rain with Thunder	Light Sleet	Light Snow	Light Snow / Freezing Rain	Light Snow / Windy	Light Snow and Sleet	Light Snow and Sleet / Windy	Mostly Cloudy	Mostly Cloudy / Windy	Partly Cloudy	Partly Cloudy / Windy	Rain	Rain / Freezing Rain	Rain / Freezing Rain / Windy	Rain / Windy	Rain and Sleet	Rain and Sleet / Windy	Rain and Snow	Rain and Snow / Windy	Shallow Fog	Sleet	Snow	Snow / Windy	Snow and Sleet	T-Storm	Thunder	Thunder / Windy	Unknown Precipitation	Wintry Mix]

one-hot weekday includes [0	1	2	3	4	5	6]


# Run 
platform windows 11

cd ./ProSTformer

run train.bat file

# visualization
cd ./ProSTformer

python visure.py
