# %%

#Predict the nightly price of AirBnB listings in London based on their characteristics.
#对每个characteristic,做一个夜间价格预测
#所以对于数据分类: 一个是characteristic的分类, 一个是夜间价格的分类. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



challenge = pd.read_csv("/Users/wangjuzheng/Downloads/2020-08-24-listings.csv.gz", dtype="unicode")
workinglist = challenge[["id","price","property_type","room_type","accommodates","bathrooms_text","bedrooms","beds","amenities"]]
missing_value=workinglist.isnull().sum()
is_entire = workinglist["room_type"] == "Entire home/apt"
is_hotel = workinglist["room_type"] == "Hotel room"
is_private = workinglist["room_type"] == "Private room"
is_shared = workinglist["room_type"] == "Shared room"

workinglist["room_type_entire"] = is_entire 
workinglist["room_type_hotel"] = is_hotel
workinglist["room_type_private"] = is_private
workinglist["room_type_shared"] = is_shared

x = workinglist[["room_type_entire", "room_type_hotel", "room_type_private", "room_type_shared"]]
y = workinglist["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse= mean_squared_error(y_test, y_pred)
print("this",mse)


  
    


    




# %%
