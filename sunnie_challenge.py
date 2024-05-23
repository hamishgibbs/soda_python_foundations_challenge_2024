#%%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load dataset
listings = pd.read_csv("2020-08-24-listings.csv.gz")
listings

# %%
# Do some exploratory analysis to understand the data.
n_bedrooms_missing = listings['bedrooms'].isnull().sum()
n_accommodates_missing = listings['accommodates'].isnull().sum()
n_price_missing = listings['price'].isnull().sum()
n_neighbourhood_missing = listings['neighbourhood_cleansed'].isnull().sum()
print(f"Missing values in 'bedrooms': {n_bedrooms_missing:,} ({n_bedrooms_missing / listings['bedrooms'].shape[0]:.2%})")
print(f"Missing values in 'price': {n_price_missing:,} ({n_price_missing / listings['price'].shape[0]:.2%})")
print(f"Missing values in 'accommodates': {n_accommodates_missing:,} ({n_accommodates_missing / listings['accommodates'].shape[0]:.2%})")
print(f"Missing values in 'neighbourhood_cleansed': {n_neighbourhood_missing:,} ({n_neighbourhood_missing / listings['price'].shape[0]:.2%})")


# %%
#____________________________________________________________________________________________________________________________________________________________________________________

#%%
### Method 1 - (Loop) In each neighbourhood, bedrooms, accommodates - MAPE 9.83%
# Get the number of unique boroughs
boroughs = listings['neighbourhood_cleansed'].unique()
boroughs_count = listings['neighbourhood_cleansed'].nunique()
print(f"There are {boroughs_count} unique neighbourhoods, they are: {', '.join(boroughs)}")

#%%
# Initialize a dictionary to store results
results_dict = {}
mape_list = []
coefficients_list = []

# Loop through each borough
for borough in boroughs:
    print(f"Processing borough: {borough}")

    # Restrict listings to the current borough
    listings_borough = listings[listings["neighbourhood_cleansed"] == borough]
    
    # Impute missing values in the 'bedrooms' column
    imputer = SimpleImputer(strategy='median')
    listings_borough['bedrooms'] = imputer.fit_transform(listings_borough[['bedrooms']])
    
    # Define features and target
    X = listings_borough[['bedrooms', 'accommodates']]
    y = listings_borough['price']
    # Convert the 'price' variable to log scale:
    y = np.log(listings_borough['price'] + 1e-1)

    # Split the dataset into training and evaluation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose and fit a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the coefficients
    coefficients = model.coef_
    
    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_percentage = mape * 100
    print(f"Mean Absolute Percentage Error for {borough}: {mape_percentage:.2f}%")
    
    # Store the results
    results_dict[borough] = {
        'model': model,
        'mape': mape,
        'coefficients': coefficients,
        'y_test': y_test,
        'y_pred': y_pred
    }

    # Append MAPE to the list
    mape_list.append({'Borough': borough, 'MAPE (%)': mape_percentage})

    # Append coefficients to the list
    coefficients_list.append({'Borough': borough, 'Coefficients': coefficients})


    # Visualize / communicate the results
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df = results_df.reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(f'Actual vs. Predicted Prices for {borough}')
    plt.xlabel('Actual Price (log$)')
    plt.ylabel('Predicted Price (log$)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect predictions
    plt.grid(True)
    plt.show()

#%%
# Calculate the overall mean MAPE
mean_mape = mape_df['MAPE (%)'].mean()
print(f"\nOverall Mean Absolute Percentage Error (MAPE): {mean_mape:.2f}%")

# Create a DataFrame for the MAPE table
mape_df = pd.DataFrame(mape_list)
print("\nMean Absolute Percentage Error (MAPE) for all Boroughs:")
print(mape_df)

# Create a DataFrame for the coefficients table
coefficients_df = pd.DataFrame(coefficients_list)
print("\nCoefficients for all Boroughs:")
print(coefficients_df)


# %%
#____________________________________________________________________________________________________________________________________________________________________________________


# %%
### Method 2 - Linear regression: Neighbourhood, bedrooms, accommodates - MAPE 9.92%

# Impute missing values in the 'bedrooms' column
imputer = SimpleImputer(strategy='median')
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])

#convert neighbourhood into dummy variables
pd.get_dummies(listings, columns=['neighbourhood_cleansed'])
neighbourhood_dummies = pd.get_dummies(listings['neighbourhood_cleansed'], prefix='neighbourhood')
neighbourhood_dummies

# %%
# Split the dataset into training / evaluation datasets.
# Combine dummy variables with other numerical features
X = pd.concat([neighbourhood_dummies, listings[['bedrooms', 'accommodates']]], axis=1)
# Define the target variable
y = listings['price']
# Convert the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)

model.coef_

#%%
# Evaluate the model performance.
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mape_percentage = mape * 100
print(f"Mean Absolute Percentage Error: {mape_percentage:.2f}%")

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
#____________________________________________________________________________________________________________________________________________________________________________________



# %%
### Method 3 - Linear regression: Neighbourhood - MAPE 13.53%
# Split the dataset into training / evaluation datasets.
# Combine dummy variables with other numerical features
X = neighbourhood_dummies
# Define the target variable
y = listings['price']
# Convert the 'price' variable to log scale:
y = np.log(listings['price'] + 1e-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
# Choose a regression to model the price of AirBnBs.
# Fit the regression model.
model = LinearRegression()
model.fit(X_train, y_train)

model.coef_

#%%
# Evaluate the model performance.
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mape_percentage = mape * 100
print(f"Mean Absolute Percentage Error: {mape_percentage:.2f}%")

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
