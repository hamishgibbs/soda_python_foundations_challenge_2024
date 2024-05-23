#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
#%%
# Read the dataset with pandas.
listings = pd.read_csv("2020-08-24-listings.csv")

# %%
# Try restricting the listings to "Entire apartment":
listings = listings[listings["property_type"] == "Entire apartment"]
# %%
listings
# %%
n_price_missing = listings['price'].isnull().sum()
# %%
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
# %%
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
imputer = SimpleImputer(strategy='median')
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])
# %%
X = listings[['bedrooms', 'accommodates', 'number_of_reviews_ltm']]
y = listings['price']

# %%
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# %%
reg = LinearRegression().fit(X, y)
reg.score(X, y)
# %%
y_pred = regressor.predict(X_test)
# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# %%
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
# %%
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
listings
# %%