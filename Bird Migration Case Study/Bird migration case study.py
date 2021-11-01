# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:20:23 2021

@author: Ali
"""
"""Import bird data from csv"""
import pandas as pd

birddata = pd.read_csv("bird_tracking.csv")

import matplotlib.pyplot as plt
import numpy as np

"""Start by isolating the data for one bird, Eric, and plotting his longitude and latitude as an xy plot"""
ix = birddata.bird_name == "Eric"

x,y = birddata.longitude[ix], birddata.latitude[ix]

plt.figure(figsize = (7,7))
plt.plot(x,y,".")

""" Now inlcude all of the birds' data and make a new version of that plot"""
bird_names = pd.unique(birddata.bird_name)

for bird_name in bird_names:
    
    ix = birddata.bird_name == bird_name

    x,y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x,y,".", label = bird_name)



plt.plot(x,y,".")
plt.xlabel("longitude")
plt.ylabel("Longitude")
plt.legend(loc = "lower right")

"""Now let's look at just Eric's average speed, and demonstrate two different ways to deal with NaNs in the data"""
ix = birddata.bird_name == "Eric"
speed = birddata.speed_2d[ix]

"""find NaNs in speed array, and only use numbers for histogram"""
ind = np.isnan(speed)

"""Now plot the cleaned data as a histogram of Eric's speeds during the data collection period"""
plt.figure(figsize = (8,4))
speed = birddata.speed_2d[birddata.bird_name == "Eric"]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins =np.linspace(0.30,20), normed =True)
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")

"""plotting the same data using pandas, which will deal with NaNs for you"""

birddata.speed_2d.plot(kind="hist", range = [0,30])
plt.xlabel("2d speed")

"""Now let's figure out Eric's average daily speed, but first we need to convert the date/time information to the datetime format"""
import datetime

date_str = birddata.date_time[0]

datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")

timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
                      (birddata.date_time.iloc[k][:-3],"%Y-%m-%d %H:%M:%S"))
        
birddata["timestamp"]= pd.Series(timestamps, index = birddata.index)

"""Create an elapsed time column that will indicate the number of days since the start of data collection"""
times = birddata.timestamp[birddata.bird_name =="Eric"]
elapsed_time = [time-times[0] for time in times]

elapsed_time[1000]/datetime.timedelta(days = 1)
 
"""Plot Eric's speed by elapsed day"""
plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time (days)")

""" Now let's figure out Eric's daily mean speed and plot it as mean speed by day since the start of observation"""
data = birddata[birddata.bird_name == "Eric"]
time = data.timestamp
elapsed_time = [time -times[0] for time in times]
elapsed_days = np.array(elapsed_time)/datetime.timedelta(days = 1)


next_day = 1
inds = []
daily_mean_speed = []

for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else: 
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []

plt.figure(figsize = (8,6))    
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean Speed (m/s)")


"""Now we're going to use cartopy to make a mercator projected map of the flight paths of all three birds"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize = (10,10))   
ax = plt.axes(projection = proj)
ax.set_extent((-25.0,20.0,52.0,10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle = ':')

for name in bird_names:
    
    ix = birddata["bird_name"] == name

    x,y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x,y,".", transform = ccrs.Geodetic(),label = name)

plt.legend(loc= "upper left")


