#!/usr/bin/env python
# coding: utf-8

# # Level - 1

# ## Task - 1

# ### Import needed libraries

# In[43]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ### Loading the data set

# In[100]:


df=pd.read_csv(r"Documents\capgemini_dataset.csv")
df


# In[101]:


df.head()


# ## Task: Data Exploration and Preprocessing

# ### Explore the dataset and identify the number of rows and columns.

# In[102]:


df.columns


# In[103]:


df.shape


# In[104]:


df.isnull().sum()


# In[49]:


df.dtypes


# In[50]:


df.info()


# ### Check for missing values in each column and handle them accordingly.

# In[51]:


df.duplicated().sum()


# In[52]:


df[df.duplicated()]


# ### Perform data type conversion if necessary.Analyze the distribution of the target variable("Aggregate rating") and identify any class imbalances.

# In[53]:


df["Aggregate rating"].value_counts()


# In[54]:


df["Rating text"].value_counts()


# ## Task - 2

# ## Task: Descriptive Analysis

# ### Calculate basic statistical measures (mean, median, standard deviation, etc.) for numerical columns.

# In[55]:


df.describe()


# ### Explore the distribution of categorical variables like "Country Code" , "City" , and "Cuisines".

# In[56]:


country_freq = df['Country Code'].value_counts()
city_freq = df['City'].value_counts()
df['Cuisines'] = df['Cuisines'].str.split(',')
df = df.explode('Cuisines')
cuisine_freq = df['Cuisines'].value_counts()


# In[57]:


plt.figure(figsize=(10,4))
country_freq.plot(kind="bar")
plt.title("Top 10 country code")
plt.xlabel("country code")
plt.ylabel("count")
plt.show()


# In[58]:


y=df["City"].value_counts().head(10)
y


# In[59]:


plt.figure(figsize=(10,6))
y.plot(kind="barh")
plt.title("Top city")
plt.xlabel("city")
plt.ylabel("count")
plt.show()


# In[60]:


z=df["Cuisines"].value_counts().head(10)
z


# In[61]:


z= df['Cuisines'].str.split(', ', expand=True).stack().value_counts().head(10)
plt.figure(figsize=(10, 6))
z.plot(kind='bar')
plt.title('top 10 Cuisines')
plt.xlabel('Cuisine')
plt.ylabel('Count')
plt.show()


# ### Identify the top cuisines and cities with the highest number of restaurants.

# In[62]:


print("Top 10 cuisines with the highest number of restaurants:")
print(cuisine_freq.head(10))
print("\n")
print("\nTop 10 cities with the highest number of restaurants:")
print(city_freq.head(10))


# ## Task - 3

# ## Task: Geospatial Analysis

# ### Visualize the locations of restaurants on a map using latitude and longitude information.

# In[63]:



correlation = df[['Latitude', 'Longitude', 'Aggregate rating']].corr().round(2)
print("\nCorrelation between Latitude, Longitude, and Aggregate rating:")
print("\n")
print(correlation)
print("\n")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm',fmt="0.2f")
plt.title('Correlation between Latitude, Longitude, and Aggregate rating')
plt.show()


# ## Determine the percentage of restaurants offering table booking and online delivery

# In[64]:


a= (df['Has Table booking'].value_counts(normalize=True) * 100).round(2)
b= (df['Has Online delivery'].value_counts(normalize=True) * 100).round(2)
print("Percentage of restaurants offering table booking:")
print(a)
print("\nPercentage of restaurants offering online delivery:")
print(b)


# ## Compare the average ratings of restaurants with table booking and those without

# In[65]:


c = df[df['Has Table booking'] == 'Yes']['Aggregate rating'].mean()
d = df[df['Has Table booking'] == 'No']['Aggregate rating'].mean()

print("Average rating of restaurants with table booking :", c)
print("\n")
print("Average rating of restaurants without table booking:", d)


# ## Analyze the availability of online deliveryamong restaurants with different price ranges.

# In[66]:


e = df['Price range'].unique()
e


# In[67]:


for prive_range in e:
    restaurants_in_prive_range = df[df['Price range'] == prive_range]
    online_delivery_percentage = (df['Has Online delivery'].value_counts(normalize=True) * 100).round(2)
    online_delivery_by_price_range = online_delivery_percentage

print("\nOnline delivery availability by price range:")
for price_range, percentage in online_delivery_by_price_range.items():
    print(f"Price Range :" ,price_range)
    print(percentage)


# # Level - 2
# 
#     

# ## Task - 1

# ## Task: Table Booking and Online Delivery

# ### Determine the percentage of restaurants that offer table booking and online delivery.

# In[68]:


f = (df['Has Table booking'].value_counts(normalize=True) * 100).round(2)
g = (df['Has Online delivery'].value_counts(normalize=True) * 100).round(2)
print("Percentage of Restaurants Offering Table Booking:") 
print(f)
print("\n")
print("\nPercentage of Restaurants Offering Online Delivery:")
print(g)


# ### Compare the average ratings of restaurants with table booking and those without.

# In[69]:


i = df.loc[df['Has Table booking'] == 'Yes', 'Aggregate rating'].mean()
j = df.loc[df['Has Table booking'] == 'No', 'Aggregate rating'].mean()
print("Average rating of restaurent with table booking : ",i)
print("Average rating of restaurent without table booking : ",j)


# ### Analyze availability of online delivery among restaurants with different price ranges

# In[70]:


k = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True) * 100
k


# In[71]:


df['Price range'].value_counts()


# ## Task - 2

# ### Determine the most common price range among all the restaurants.

# In[72]:



a = df['Price range'].value_counts()
print(a)
b = a.idxmax()  
c = a.max() 
print(f"The most common price range among all the restaurants is:",b)
print(f"It appears {c} times.")


# ### Calculate the average rating for each price range.

# In[73]:


x = df.pivot_table(index='Price range', values='Aggregate rating', aggfunc='mean')
print("Average rating for each price range:")
print(x)


# ### Identify the color that represents the highest average rating among different price ranges.

# In[74]:


def col_with_high_avg_rating(df):
    x = df.groupby(['Price range', 'Rating color'])['Aggregate rating'].mean()

    y = x.reset_index()
    z = y['Aggregate rating'].idxmax()
    high_color = y.loc[z, 'Rating color']
    high_price_range = y.loc[z, 'Price range']
    high_rating = y.loc[z, 'Aggregate rating']
    print(f"highest average rating of :",high_rating)
    print('high color :',high_color)
    print("price range :",high_price_range)
    return {
        'Rating color': high_color,
        'price_range': high_price_range,
        'rating': high_rating
    }
result = col_with_high_avg_rating(df)


# ## Task - 3

# ### Extract additional features from the existing columns, such as the length of the restaurant name or address.

# In[75]:


def extract_features(df):
    df['Restaurant Name Length'] = df['Restaurant Name'].str.len()
    df['Address Length'] = df['Address'].str.len()
    print("Extracted features:")
    print(df[['Restaurant Name Length', 'Address Length']].head())
    return df
df = extract_features(df)


# ### Create new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables.

# In[76]:


def encode_cat_var(df):
    df['Has Table Booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Encoded features:")
    print(df[['Has Table Booking', 'Has Online delivery']].head())
    return df
df = encode_cat_var(df)


# # Level - 3

# ## Task - 1

# ### Build a regression model to predict the aggregate rating of a restaurant based of available features.

# In[77]:


def build_regression_model(df):
    features_to_keep = ['Price range', 'Rating color', 'Restaurant Name', 'Address',
                        'Has Table booking', 'Has Online delivery']
    target = 'Aggregate rating'
    df = df.dropna(subset=[target])
    df['Restaurant Name Length'] = df['Restaurant Name'].str.len()
    df['Address Length'] = df['Address'].str.len()
    X = df[features_to_keep + ['Restaurant Name Length', 'Address Length']]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    categorical_features = ['Price range', 'Rating color', 'Has Table booking', 'Has Online delivery']
    numerical_features = ['Restaurant Name Length', 'Address Length']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")
    return pipeline
model = build_regression_model(df)


# In[79]:


df


# ### Split the dataset into training and testing sets and evaluate the model's performance using appropriate metrics.

# In[80]:


def evaluate_regression_model(df, target_column, test_size=0.2, random_state=42):
    df = df.dropna(subset=[target_column])
    df['Restaurant Name Length'] = df['Restaurant Name'].str.len()
    df['Address Length'] = df['Address'].str.len()
    features_to_use = ['Price range', 'Rating color', 'Restaurant Name Length', 'Address Length',
                       'Has Table booking', 'Has Online delivery']
    X = df[features_to_use]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    categorical_features = ['Price range', 'Rating color', 'Has Table booking', 'Has Online delivery']
    numerical_features = ['Restaurant Name Length', 'Address Length']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")
    return pipeline, {"RMSE": rmse, "MAE": mae, "R2": r2}
target_column = 'Aggregate rating'
model, metrics = evaluate_regression_model(df, target_column)


# ### Experiment with different algorithms (e.g., linear regression, decision trees, random forest) and compare their performance.

# In[81]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def experiment_with_algorithms(df, target_column, test_size=0.2, random_state=42):
    df = df.dropna(subset=[target_column])
    df['Restaurant Name Length'] = df['Restaurant Name'].str.len()
    df['Address Length'] = df['Address'].str.len()
    features_to_use = ['Price range', 'Rating color', 'Restaurant Name Length', 'Address Length',
                       'Has Table booking', 'Has Online delivery']
    X = df[features_to_use]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    categorical_features = ['Price range', 'Rating color', 'Has Table booking', 'Has Online delivery']
    numerical_features = ['Restaurant Name Length', 'Address Length']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(random_state=random_state)
    }
    model_metrics = {}
    for algorithm_name, model in algorithms.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_metrics[algorithm_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        print(f"\n{algorithm_name} Performance:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared: {r2:.2f}")
    return model_metrics
target_column = 'Aggregate rating'
model_metrics = experiment_with_algorithms(df, target_column)


# ## Task - 2

# ### Analyze the relationship between the type of cuisine and the restaurant's rating.

# In[82]:


df = df.dropna(subset=['Aggregate rating'])
df['Cuisines'] = df['Cuisines'].str.split(',')
df = df.explode('Cuisines')
cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=cuisine_ratings.index, y=cuisine_ratings.values, palette='viridis')
plt.title('Mean Aggregate Rating by Cuisine Type')
plt.xlabel('Cuisine Type')
plt.ylabel('Mean Aggregate Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### Identify the most popular cuisines among customers based on the number of votes.

# In[83]:


df = df.dropna(subset=['Votes'])
df['Cuisines'] = df['Cuisines'].str.split(',')
df = df.explode('Cuisines')  
cuisine_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
cuisine_votes.head(10).plot(kind='bar', color='mediumseagreen')
plt.title('Top 10 Most Popular Cuisines Based on Number of Votes')
plt.xlabel('Cuisine')
plt.ylabel('Total Votes')
plt.xticks(rotation=45, ha='right')
plt.show()
print("Top 10 most popular cuisines based on the number of votes:")
print(cuisine_votes.head(10))


# ### Determine if there are any specific cuisines that tend to receive higher ratings.

# In[105]:


df = df.dropna(subset=['Aggregate rating'])
df['Cuisines'] = df['Cuisines'].str.split(',')
df = df.explode('Cuisines')
cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=cuisine_ratings.index, y=cuisine_ratings.values, palette='viridis')
plt.title('Average Aggregate Rating by Cuisine Type')
plt.xlabel('Cuisine Type')
plt.ylabel('Average Aggregate Rating')
plt.xticks(rotation=45, ha='right')
plt.show()
print("Top 10 cuisines with the highest mean ratings:")
print(cuisine_mean_ratings.head(10))


# ## Task - 3

# ### Create visualizations to represent the distribution of ratings using different charts (histogram, bar plot, etc.).

# In[85]:


df = df.dropna(subset=['Aggregate rating'])
plt.figure(figsize=(10, 6))
sns.histplot(df['Aggregate rating'], bins=10, kde=True, color='steelblue')
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Aggregate rating'], color='mediumseagreen')
plt.title('Box Plot of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.show()
df['Rounded Rating'] = df['Aggregate rating'].round()

plt.figure(figsize=(10, 6))
sns.countplot(x='Rounded Rating', data=df, palette='viridis')
plt.title('Count of Restaurants by Rounded Ratings')
plt.xlabel('Rounded Aggregate Rating')
plt.ylabel('Count')
plt.show()


# ### Compare the average ratings of different cuisines or cities using appropriate visualizations.

# In[86]:


df = df.dropna(subset=['Aggregate rating'])
df['Cuisines'] = df['Cuisines'].str.split(',')
df = df.explode('Cuisines') 
cuisine_mean_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
city_mean_ratings = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=cuisine_mean_ratings.index, y=cuisine_mean_ratings.values, palette='coolwarm')
plt.title('Mean Aggregate Rating by Cuisine Type')
plt.xlabel('Cuisine')
plt.ylabel('Mean Aggregate Rating')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.figure(figsize=(12, 8))
sns.barplot(x=city_mean_ratings.index, y=city_mean_ratings.values, palette='coolwarm')
plt.title('Mean Aggregate Rating by City')
plt.xlabel('City')
plt.ylabel('Mean Aggregate Rating')
plt.xticks(rotation=45, ha='right')
plt.show()
print("Top 10 cuisines with the highest mean ratings:")
print(cuisine_mean_ratings.head(10))

print("\nTop 10 cities with the highest mean ratings:")
print(city_mean_ratings.head(10))


# ### Visualize the relationship between various features and the target variable to gain insights.

# In[87]:



df = pd.get_dummies(df, columns=['City', 'Cuisines'], drop_first=True)
target = 'Aggregate rating'
features = ['Votes', 'Price range', 'Has Table booking', 'Has Online delivery']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df[target], color='steelblue')
    plt.title(f'Relationship between {feature} and {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()
categorical_features = ['City', 'Cuisines']
for feature in categorical_features:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=feature, y=target, data=df, palette='coolwarm')
    plt.title(f'Aggregate Rating by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Aggregate Rating')
    plt.xticks(rotation=45, ha='right')
    plt.show()
sns.pairplot(df, vars=features + [target], diag_kind='kde')
plt.show()


# In[ ]:





# In[ ]:




