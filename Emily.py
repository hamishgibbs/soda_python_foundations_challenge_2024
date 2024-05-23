#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
#%%

# Read the dataset with pandas.
listings = pd.read_csv("2020-08-24-listings.csv")
# %%
listings['property_type']
# %%
# Try restricting the listings to "Entire apartment":
listings = listings[listings["property_type"] == "Entire apartment"]
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
#%%
listings['bedrooms'].hist(bins=50)
#%%
listings['accommodates'].hist(bins=50)
# %%
#missing values will be replaced with the median value of the respective column.
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
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
# %%
# Try a different model:
# model = DecisionTreeRegressor(random_state=0)
# model.fit(X_train, y_train)
#%%
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

# Do some exploratory analysis to understand the data.
n_beds_missing = listings['beds'].isnull().sum()
print(f"Missing values in 'beds': {n_beds_missing:,} ({n_beds_missing / listings['beds'].shape[0]:.2%})")

# %%
listings['beds'].hist(bins=50)
# %%
#missing values will be replaced with the median value of the respective column.
imputer = SimpleImputer(strategy='mean')
listings['beds'] = imputer.fit_transform(listings[['beds']])

# %%
X = listings[['beds', 'accommodates']]  # Using only bedrooms to predict the price
y = listings['price']
# Try converting the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
# %%
# Try a different model:
# model = DecisionTreeRegressor(random_state=0)
# model.fit(X_train, y_train)
#%%
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
#number_of_reviews

# Do some exploratory analysis to understand the data.
n_number_of_reviews_missing = listings['number_of_reviews'].isnull().sum()
print(f"Missing values in 'number_of_reviews': {n_number_of_reviews_missing:,} ({n_number_of_reviews_missing / listings['number_of_reviews'].shape[0]:.2%})")

# %%
listings['number_of_reviews'].hist(bins=50)
# %%
#missing values will be replaced with the median value of the respective column.
imputer = SimpleImputer(strategy='mean')
listings['number_of_reviews'] = imputer.fit_transform(listings[['number_of_reviews']])

# %%
X = listings[['number_of_reviews', 'accommodates', 'beds']]  # Using only bedrooms to predict the price
y = listings['price']
# Try converting the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
# %%
# Try a different model:
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)
#%%
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
#amenities# Do some exploratory analysis to understand the data.
n_amenities_missing = listings['amenities'].isnull().sum()
print(f"Missing values in 'amenities': {n_amenities_missing:,} ({n_amenities_missing / listings['amenities'].shape[0]:.2%})")

# %%
#availability_30
# Do some exploratory analysis to understand the data.
n_availability_30_missing = listings['availability_30'].isnull().sum()
print(f"Missing values in 'availability_30': {n_availability_30_missing:,} ({n_availability_30_missing / listings['availability_30'].shape[0]:.2%})")

# %%
