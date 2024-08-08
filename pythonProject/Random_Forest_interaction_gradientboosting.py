# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:41:40 2024

@author: Peter
"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor


#PEARSON CORRELATION BETWEEN HOMERUNS AND WIND

df = pd.read_csv('C:\\Users\\Peter\\Desktop\\Final stuff\\merged_file.csv')

wind_features = df[['Windspeed', 'Wind Direction', 'Home Runs']]

windspeed = wind_features['Windspeed'].corr(wind_features['Home Runs'])
wind_direction = wind_features['Wind Direction'].corr(wind_features['Home Runs'])
direction_diff = df['Direction Difference(WindDirection - Stadium Direction)'].corr(df['Home Runs'])


print("Correlation between Direction Difference and Home Runs:", direction_diff)
print("Correlation between Windspeed and Home Runs:", windspeed)
print("Correlation between Wind Direction and Home Runs:", wind_direction)

##########################################################################################################################
##########################################################################################################################


# INTERACTION OF FEATURES

pairs = [
    ('Temperature', 'Humidity'),
    ('Temperature', 'Windspeed'),
    ('Temperature', 'Air Pressure'),
    ('Windspeed', 'Direction Difference(WindDirection - Stadium Direction)'),
    ('Dewpoint', 'Humidity'),
    ('Air Pressure', 'Elevation(meters)'),
    ('Dewpoint', 'Windspeed'),
    ('Temperature', 'Elevation(meters)'),
    ('Elevation(meters)', 'Windspeed'),
    ('Elevation(meters)', 'Dewpoint'),
    ('Elevation(meters)', 'Humidity')
]


for feature1, feature2 in pairs:
    interaction_feature_name = f'{feature1} * {feature2}'
    df[interaction_feature_name] = df[feature1] * df[feature2]


interaction_of_features = [f'{f1} * {f2}' for f1, f2 in pairs]

interaction_of_features.append('Slugging %')

interaction_correlation_matrix = df[interaction_of_features].corr()
correlation = interaction_correlation_matrix['Slugging %'].sort_values(ascending=False)

print(correlation)


# ##########################################################################################################################
# ##########################################################################################################################


# # RANDOM FOREST MODEL

features = [
    'Temperature', 'Humidity', 'Windspeed', 'Month', 'Is_Weekend',
    'Dewpoint', 'Wind Direction', 'Stadium Direction', 
    'Direction Difference(WindDirection - Stadium Direction)',
    'Air Pressure', 'Elevation(meters)'
]

X = df[features]
y = df['Slugging %']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200, 300, 500]  # Different values for the number of trees
}


rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)


grid_search.fit(X_train, y_train)

best_n_estimators = grid_search.best_params_['n_estimators']
print("Best number of trees (n_estimators):", best_n_estimators)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
print()
print()


##########################################################################################################################
##########################################################################################################################

# GRADIENT BOOSTING



label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Team'] = label_encoder.fit_transform(df['Team'])
df['Opponent'] = label_encoder.fit_transform(df['Opponent'])
df['Day_of_Week'] = label_encoder.fit_transform(df['Day_of_Week'])


X = df[['Temperature', 'Dewpoint', 'Humidity', 'Wind Direction', 'Stadium Direction', 
        'Direction Difference(WindDirection - Stadium Direction)', 'Windspeed', 
        'Air Pressure', 'Elevation(meters)']]
y = df['Slugging %']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
