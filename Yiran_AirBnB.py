#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction import text 
import matplotlib.pyplot as plt
import numpy as np
import re


#%%
# subsetting the variables i want to use
data = pd.read_csv('/Users/yangyiran/Desktop/SODA_PythonCourse/2020-08-24-listings.csv')
variables = ['id','name','description','host_since','room_type','accommodates','bedrooms','beds','minimum_nights','maximum_nights','availability_30','availability_60','availability_90','instant_bookable', 'calculated_host_listings_count', 'price']
listings = data[variables]
listings

#%%
# Data exploration: check for missing values in each variable used
for variable in variables:
    n_missing = listings[variable].isnull().sum()
    print(f"Missing values in {variable}: {n_missing:,} ({n_missing / listings[variable].shape[0]:.2%})")

# Data exploration: distribution of continuous variables
# plot histograms of numeric variables
numeric_variables = listings.select_dtypes(include=['number']).drop(columns=['id']) # select all numeric variables
for numeric_variable in numeric_variables: 
        plt.figure() # create histograms
        listings[numeric_variable].hist(bins=50)
        plt.title(f'Histogram of {numeric_variable}')  # Set title
        plt.show() 

# Data exploration: tabulate frequencies of categorical variables
room_type_counts = listings['room_type'].value_counts() # tabulates the types of properties
instant_bookable_percentage = (listings['instant_bookable'].value_counts()/len(listings))*100 # percentage of properties that can be booked without host confirmation
print(room_type_counts)
print(instant_bookable_percentage)


#%%
# preparing data for use in regression
# imputation for missing values
imputer = SimpleImputer(strategy='mean') # using mean because bedrooms and beds are both positively skewed
listings['bedrooms'] = imputer.fit_transform(listings[['bedrooms']])
listings['beds'] = imputer.fit_transform(listings[['beds']])

# transform host_since to duration the host has been a host
listings['host_since'] = pd.to_datetime(listings['host_since']) # convert to date/time type
reference_date = pd.Timestamp.today() # set reference date to today
listings['years_since'] = (reference_date - listings['host_since']) / pd.Timedelta(days=365) # calculate difference between today's date and host_since
listings['years_since'] = listings['years_since'].astype(float) # convert to float

# room_type


# Choosing predictor variables
        # instant_bookable is not used because literally all the listings are instantly bookable
        # 

#%%
# Test-training split
X = listings[['years_since','accommodates','bedrooms','beds','minimum_nights','maximum_nights','availability_30','availability_60','availability_90', 'calculated_host_listings_count']]
y = np.log(listings['price'] + 1e-1) # log scale is applied to price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit different regression models
linear_reg = LinearRegression()
decision_tree_reg = DecisionTreeRegressor()
random_forest_reg = RandomForestRegressor()
estimators = [linear_reg, decision_tree_reg, random_forest_reg]

for estimator in estimators: 
    estimator.fit(X_train,y_train)
    # Cross-validation
    scores = cross_val_score(estimator, X, y, cv=5)
    # MAPE
    y_pred = estimator.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # print validation results
    print(str(estimator),"%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()),"Mean Absolute Percentage Error:", mape)
    
#%%
# Visualising results
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
# Which boroughs have the most expensive AirBnBs?
x = data['neighbourhood_cleansed']
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer) # initialise tf-idf vectoriser 
X_text = tfidf_vectorizer.fit_transform(x)
random_forest_reg.fit(X_text,listings['price'])
coefficients = random_forest_reg.feature_importances_
expensive_boroughs = tfidf_vectorizer.get_feature_names_out()
coefficients_boroughs = pd.DataFrame({'Boroughs': expensive_boroughs, 'Coefficient': coefficients})
coefficients_sorted = coefficients_boroughs.sort_values(by='Coefficient', ascending=False)
most_expensive_boroughs = coefficients_sorted.head(10)['Boroughs'] # finds 10 words associated with most expensive boroughs 
print("Most Expensive Boroughs:")
print(most_expensive_boroughs)

# %%
