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
listings = pd.read_csv("2020-08-24-listings.csv.gz")
#%%
# Try restricting the listings to "Entire apartment":
# listings = listings[listings["property_type"] == "Entire apartment"]

entire_home_or_apt = listings[listings["room_type"] == "Entire home/apt"]
private_room = listings[listings["room_type"] == "Private room"]
hotel_room = listings[listings["room_type"] == "Hotel room"]
shared_room = listings[listings["room_type"] == "Shared room"]
# %%
# Do some exploratory analysis to understand the data.
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
n_price_missing = listings['price'].isnull().sum()
print(f"Missing values in 'bedrooms': {n_bedrooms_missing:,} ({n_bedrooms_missing / listings['bedrooms'].shape[0]:.2%})")
print(f"Missing values in 'price': {n_price_missing:,} ({n_price_missing / listings['price'].shape[0]:.2%})")
print(f"Missing values in 'accommodates': {n_accommodates_missing:,} ({n_accommodates_missing / listings['accommodates'].shape[0]:.2%})")
#%%
listings['beds'].hist(bins=50)
#%%
listings['bedrooms'].hist(bins=50)
#%%
listings['bathrooms_text'].hist(bins=50) 
#these histograms have been done to see the distribution and if you want to set the missing information as the median or something like that
#as an assumption, like he's done below with the imputer command, or whether you want to just remove the rows and columns with any missing data
# %%
#Set the median as the missing data for bedrooms and beds, remove any missing data from the listings for the bathrooms
imputer = SimpleImputer(strategy='median')
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])
listings["beds"] = imputer.fit_transform(listings[["bedrooms"]])

listings.dropna(axis=0, subset="bathrooms_text", inplace = True)
# %%
# Split the dataset into training / evaluation datasets.

#looking at the some of the x variables, it may make sense to log them too
#listings["host_total_listings_count"] = np.log(listings['host_total_listings_count'] + 1e-1)

#X = listings[['host_total_listings_count', 'host_identity_verified', "room_type", "bathrooms_text", "bedrooms", "beds", "number_of_reviews_l30d", "calculated_host_listings_count"]]
X = listings[['host_total_listings_count', "beds", "number_of_reviews_l30d", "calculated_host_listings_count"]]
y = listings['price']
# listings["beds"].corr(listings["bedrooms"]) --> beds and bedrooms have a 1 correlation
# Try converting the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
#%%
# Try a different model:
#model = DecisionTreeRegressor(random_state=0)
#model.fit(X_train, y_train)
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
