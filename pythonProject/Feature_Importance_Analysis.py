# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:53:48 2024

@author: Peter
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Peter\\Desktop\\Final stuff\\merged_file.csv')


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


feature_importances = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()
