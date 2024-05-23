#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
#%%
# Read the dataset with pandas.
listings = pd.read_csv("/Users/sheann/Desktop/python/2020-08-24-listings.csv")
# %%
listings['neighbourhood_cleansed'].unique()
print(listings['neighbourhood_cleansed'])
#%%
listings['neighbourhood_cleansed'].value_counts()
# %%
# assigning each neighbourhood to a zone
neighbourhood = listings['neighbourhood_cleansed']
zone_1 = neighbourhood.isin(["City of London", "Camden", "Hackney", "Islington", "Kensington and Chelsea", "Lambeth", "Southwark", "Tower Hamlets", "Wandsworth", "Westminster"])
zone_2 = neighbourhood.isin(["Hammersmith and Fulham", "Lewisham", "Newham", "Brent", "Ealing", "Greenwich", "Hounslow"])
zone_3 = neighbourhood.isin(["Haringey", "Barnet", "Bromley", "Croydon", "Merton", "Richmond upon Thames", "Waltham Forest"])
zone_4 = neighbourhood.isin(["Redbridge", "Enfield", "Barking and Dagenham", "Bexley", "Sutton"])
zone_6 = neighbourhood.isin(["Havering", "Hillingdon", "Harrow", "Kingston upon Thames"])
#zone_1_listings = listings[zone_1 & (listings['price'] != 0)]

listings['zone'] = 0  # Initialize 'zone' column with 0 for all rows
listings.loc[zone_1, 'zone'] = 1 
listings.loc[zone_2, 'zone'] = 2
listings.loc[zone_3, 'zone'] = 3
listings.loc[zone_4, 'zone'] = 4
listings.loc[zone_6, 'zone'] = 6

# %%
# Try restricting the listings to "Entire apartment":
listings_cleaned = listings[listings["property_type"] == "Entire apartment"].copy()
# %%
# creating new column for price per person
listings_cleaned['price_per_person'] = listings_cleaned['price'] / listings_cleaned['accommodates']
# %%
# Combine the text from 'description' and 'name' for analysis
listing_desc = listings_cleaned['name'].fillna('') + ' ' + listings_cleaned['description'].fillna('')
# Identify luxury vs non-luxury listings
expensive_listings = listing_desc.str.contains('luxury')  | listing_desc.str.contains('deluxe') 
listings_cleaned['expensive_listings'] = expensive_listings
#%%
# Function to extract numbers from text and convert to float
def extract_floats(text):
    if pd.isna(text):
        return None
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    return float(numbers[0]) if numbers else None

# Apply the function to the DataFrame column
listings_cleaned['bathrooms'] = listings_cleaned['bathrooms_text'].apply(extract_floats)
# deleting rows with missing values 
listings_cleaned = listings_cleaned.dropna().copy()
#%%
# check for outliers in price pp
mean_price = listings_cleaned['price_per_person'].mean()
std_price = listings_cleaned['price_per_person'].std()

# Calculate Z-scores
listings_cleaned['z_score'] = (listings_cleaned['price_per_person'] - mean_price) / std_price

# Filter out outliers (keep rows with abs(Z-score) <= 3)
listings_no_outliers = listings_cleaned[np.abs(listings_cleaned['z_score']) <= 3].copy()

# Drop the Z-score column as it's no longer needed
listings_no_outliers.drop(columns=['z_score'], inplace=True)


#%%
# Split the dataset into training / evaluation datasets.
X = listings_no_outliers[['bedrooms', 'zone', 'price_per_person', 'expensive_listings', 'bathrooms']]  
#y = zone_1_listings_cleaned['price']
# Try converting the 'price' variable to log scale:
y = np.log(listings_no_outliers['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)
#%%
model.coef_
#%%
# Evaluate the model performance.
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print("R^2 value:", r_squared)
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
