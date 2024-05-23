#Session 4 Challenge
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
#%%
plt.style.use('ggplot')
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)
plt.rcParams['figure.figsize'] = (15, 5)
# %%
listings = pd.read_csv("Data/2020-08-24-listings.csv") #when do we need to specify the dtype?
#%%
listingsc = listings.dropna() #complete case sample
# %%
#exploratory analysis
bedroomc = listingsc["bedrooms"].value_counts()
bedroomc.plot(kind="bar")
print(bedroomc)
#%%
propertyc = listingsc["property_type"]
#%%
accommsc = listingsc["accommodates"].value_counts()
accommsc.plot(kind="bar")
#%%
roomt = listingsc["room_type"]
roomt.value_counts()
#%%
#%%
price = listingsc["price"]
price[:5000].hist(bins=50)
#%%
np.log(price + 1e-1).hist(bins=50)
fprice = np.log(price + 1e-1)
#%%
neighbourhoodc = listingsc["neighbourhood_cleansed"]
bneighbourghoodc = pd.get_dummies(neighbourhoodc)
#%%
len(bneighbourghoodc)
bneighbourhoodc = bneighbourghoodc.astype(int)
#%%
#coding to central london and not, adding more variables
#bneighbourhoodc["Central"] = listingsc['neighbourhood_cleansed'].str.contains('Camden|Greenwich|Hackney|Hammersmith and Fulham|Islington|Kensington and Chelsea|Lambeth|Lewisham|Southwark|Tower Hamlets|Wandsworth|Westminster').astype(float) #classifying neighbourhoods as in Central London or Not
bneighbourhoodc["Bedrooms"]=listingsc["bedrooms"]
bneighbourhoodc["Accommodates"]=listingsc["accommodates"]
bneighbourhoodc["Number of Beds"]=listingsc["beds"]
#%%
#number of rooms
broomt = pd.get_dummies(roomt).astype(float)
#%%
finaldata = pd.concat([bneighbourhoodc, broomt], axis=1)
# %%
#Fix Method
X = finaldata 
y = fprice
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#Running Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
#testing mean absolute percentage error
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape) #0.08 (2 d.p.)
#%%
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results = results.reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect predictions
plt.grid(True)
plt.show()
# %%
evaluation = cross_validate(model, X, y)
evaluation
