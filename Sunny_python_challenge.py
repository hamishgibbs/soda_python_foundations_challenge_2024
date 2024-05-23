# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
plt.style.use('ggplot')
# %%
#read the dataset with pandas
listings = pd.read_csv("/Users/sunnyren/Desktop/python/2020-08-24-listings.csv")
# %%
listings[:10]
# %%
#see the common types of the airbnb listing
listings['neighbourhood_cleansed'].value_counts()
listings['room_type'].value_counts()
# %%
# Do some exploratory analysis to understand the data.
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
n_price_missing = listings['price'].isnull().sum()
print(f"Missing values in 'bedrooms': {n_bedrooms_missing:,} ({n_bedrooms_missing / listings['bedrooms'].shape[0]:.2%})")
print(f"Missing values in 'price': {n_price_missing:,} ({n_price_missing / listings['price'].shape[0]:.2%})")
print(f"Missing values in 'accommodates': {n_accommodates_missing:,} ({n_accommodates_missing / listings['accommodates'].shape[0]:.2%})")
# %%
listings['price'].hist(bins=50)
# %%
listings['bedrooms'].hist(bins=50)
# %%
imputer = SimpleImputer(strategy='median')
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])
# %%
# Split the dataset into training / evaluation datasets.
X = listings[['bedrooms', 'accommodates']]  # Using only bedrooms to predict the price
y = listings['price']
# Try converting the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
# %%
model.coef_
# %%
# Evaluate the model performance (very bad!).
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
# %%
# Visualize / communicate the results. 
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

#normalise price
#see missing values in price
print(listings['price'].max())
print(listings['price'].min())
# %%
listings['price'].unique()
listings['accommodates'].unique()
# %%
listings['bedrooms'].unique()
# %%
na_values = ['0', 'N/A']
# %%
listings = pd.read_csv('/Users/sunnyren/Desktop/python/2020-08-24-listings.csv', na_values=na_values, dtype={'price': float})
# %%
listings['price'].hist(bins=50)
# %%
print(listings['price'])
# %%
listings['price'].dtype
# %%
#convert price to log
listings['price_log'] = np.log(listings['price'] + 1e-1)
# %%
listings['price_log'].hist(bins=50)
# %%


#variable room type
listings['room_type'].unique()
# %%
pd.get_dummies(listings["room_type"])
# %%
room_dummies = pd.get_dummies(listings['room_type'], prefix='room')
# %%
#attaching two parts together
listings_with_dummies = pd.concat([listings, room_dummies], axis=1)
# %%
listings_with_dummies['room_Entire home/apt']

# %%
#handle the missing data
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
# %%
listings_with_dummies[['room_Entire home/apt', 'room_Hotel room', 'room_Private room', 'room_Shared room', 'bedrooms', 'accommodates']].isnull().sum()
# %%
n_bedrooms_missing = listings_with_dummies['bedrooms'].isnull().sum()
# %%
print(f"Missing values in 'bedrooms': {n_bedrooms_missing:,} ({n_bedrooms_missing / listings_with_dummies['bedrooms'].shape[0]:.2%})")
# %%
imputer = SimpleImputer(strategy='median')
listings_with_dummies['bedrooms'] = imputer.fit_transform(listings_with_dummies[['bedrooms']])
# %%
n_price_missing = listings_with_dummies['price'].isnull().sum()
# %%
print(f"Missing values in 'price': {n_price_missing:,} ({n_price_missing / listings_with_dummies['price'].shape[0]:.2%})")
# %%
y = np.log(listings_with_dummies['price'] + 1e-1)
# %%
imputer = SimpleImputer(strategy='median')
listings_with_dummies['price'] = imputer.fit_transform(listings_with_dummies[['price']])

# %%
# %%
#run the model
print(listings_with_dummies.columns)
X = listings_with_dummies[['room_Entire home/apt', 'room_Hotel room', 'room_Private room', 'room_Shared room', 'bedrooms', 'accommodates']]
y = listings_with_dummies['price']
# %%
y = np.log(listings_with_dummies['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
model = LinearRegression()
model.fit(X_train, y_train)
# %%
model.coef_
# %%
# Evaluate the model performance
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
# %%
# Visualize / communicate the results. 
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

#string operation
room_description = listings_with_dummies['description']
# %%
is_luxury = room_description.str.contains('lux')
# %%
listings_with_dummies[is_luxury] #there are missing NaN values
# %%
#check the number of missing values - 2806
n_des_missing = listings_with_dummies['description'].isnull().sum()
print(f"Missing values in 'description': {n_des_missing:,} ({n_des_missing / listings_with_dummies['description'].shape[0]:.2%})")
# %%
#change the missing values into false (not luxury)
is_notlx = ~(is_luxury & room_description.notnull())
# %%
listings_with_dummies['is_notlx'] = ~(is_luxury & room_description.notnull())
# %%
X = listings_with_dummies[['room_Entire home/apt', 'room_Hotel room', 'room_Private room', 'room_Shared room', 'bedrooms', 'accommodates','is_notlx']]
y = listings_with_dummies['price']
# %%
y = np.log(listings_with_dummies['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
# %%
# Evaluate the model performance
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
# %%

#