#%%
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
# %%
listings = pd.read_csv("./Data/2020-08-24-listings.csv.gz")
# %%
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
n_price_missing = listings['price'].isnull().sum()
n_reviews_missing = listings['number_of_reviews'].isnull().sum()
n_listing_count_missing = listings['calculated_host_listings_count'].isnull().sum()
print(f"Missing values in 'bedrooms': {n_bedrooms_missing:,} ({n_bedrooms_missing / listings['bedrooms'].shape[0]:.2%})")
print(f"Missing values in 'price': {n_price_missing:,} ({n_price_missing / listings['price'].shape[0]:.2%})")
print(f"Missing values in 'accommodates': {n_accommodates_missing:,} ({n_accommodates_missing / listings['accommodates'].shape[0]:.2%})")
print(f"Missing values in 'reviews': {n_reviews_missing:,} ({n_reviews_missing / listings['number_of_reviews'].shape[0]:.2%})")
print(f"Missing values in 'listing count': {n_listing_count_missing:,} ({n_listing_count_missing / listings['calculated_host_listings_count'].shape[0]:.2%})")
# %%
listings['price'].hist(bins=100)
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Histogram of Prices')
plt.xlim(0, 2000)
# %%
listings['accommodates'].hist(bins=20)
plt.xlabel('Number of People Accommodated')
plt.ylabel('Frequency')
plt.title('Histogram of Accommodation Capacity') 
# %%
listings['bedrooms'].hist(bins=20)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.title('Histogram of Bedrooms') 
plt.xlim(0, 10)
# %%
ultra_high_price_listings = listings[listings['price']>2000].shape[0]
high_price_listings = listings[listings['price']>1750].shape[0]
mid_high_price_listings = listings[listings['price']>1500].shape[0]
mid_price_listings = listings[listings['price']>1250].shape[0]
print(ultra_high_price_listings)
print(high_price_listings)
print(mid_price_listings)
print(mid_price_listings)
# %%
total_count = listings[listings['price']>-1000].shape[0]
print(total_count)
# %%
lower_price_listings = listings[listings['price']>1000].shape[0]
print(lower_price_listings)
other_price_listings = listings[listings['price']>150].shape[0]
print(other_price_listings)
# %%
outlier_price_listings = listings['price']>1000
listings.loc[outlier_price_listings, 'price'] = np.nan
# %%
lower_price_listings = listings[listings['price']>1000].shape[0]
print(lower_price_listings)
# %%
imputer = SimpleImputer(strategy='median')
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])
listings['price'] = imputer.fit_transform(listings[['price']])
# %%
room_type_dummies = pd.get_dummies(listings['room_type'])
listings = pd.concat([listings, room_type_dummies], axis=1)
# %%
imputer = SimpleImputer(strategy='median')
listings['beds'] = imputer.fit_transform(listings[['beds']]) 
# %%
X = listings[['bedrooms', 'accommodates', 'number_of_reviews', 'beds', 'minimum_nights', 'maximum_nights', 'calculated_host_listings_count', 'number_of_reviews_l30d', 'Entire home/apt', 'Hotel room', 'Private room', 'Shared room']]  
y = np.log(listings['price'] + 1e-1)
# %%
check_nan = listings['beds'].isnull().values.any()
print(check_nan) # hence if I add 'bed' to the regression I must first remove the NaN values and replace them with median
# %%
 
# %%
scaler = StandardScaler()
X = scaler.fit_transform(X, y)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# %%
y_prediction = model.predict(X_test)
y_prediction
# %%
score = r2_score(y_test,y_prediction)
print('r2 score is ',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))
# %%
mape = mean_absolute_percentage_error(y_test, y_prediction)
print("Mean Absolute Percentage Error:", mape) #significantly lower than the 70% in the scaffolding 
# %%
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})
results = results.reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_prediction, alpha=0.5)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  
plt.grid(True)
plt.show()
# %%
