import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# read input file
def read_files():
    df = pd.read_csv("/mnt/data/merged_file_mlbgames_weather.csv")
    return df

# Descriptive statistics
def descriptive_stats(games):
    print("Descriptive Statistics:")
    print("-----------------------")
    print(games.describe())
    print()

# Missing data analysis
def missing_data_analysis(games):
    print("Missing Data Analysis:")
    print("----------------------")
    missing_data = games.isnull().sum()
    print(missing_data)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(games.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap of Missing Data')
    plt.show()
    print()

# Visualizing distributions
def visualize_distributions(games):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(games["temp"].dropna(), kde=True, bins=30)
    plt.title('Histogram of Temperature')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=games["temp"].dropna())
    plt.title('Box Plot of Temperature')
    
    plt.show()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(games["dwpt"].dropna(), kde=True, bins=30)
    plt.title('Histogram of Dew Point')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=games["dwpt"].dropna())
    plt.title('Box Plot of Dew Point')
    
    plt.show()

# Visualizing categorical data
def visualize_categorical_data(games):
    plt.figure(figsize=(14, 7))
    
    sns.countplot(data=games, x='Home', order=games['Home'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Count of Home Teams')
    plt.show()
    
    plt.figure(figsize=(14, 7))
    
    sns.countplot(data=games, x='Visitor', order=games['Visitor'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Count of Visitor Teams')
    plt.show()

# Outlier detection
def detect_outliers(games):
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(data=games[["temp", "dwpt", "wspd", "pres"]])
    plt.title('Box Plot of Weather Features')
    plt.show()

# Main program
def main():
    games = read_files()
    
    # Calculation of slugging percentage
    games["SLG_PCT"] = (games["Singles"] + (games["Doubles"] * 2) + 
                        (games["Triples"] * 3) + (games["Home Runs"] * 4)) / games["At Bats"]
    
    # Print dimension of dataset
    gs = games.shape
    print("The dimensions of the Baseball-Weather dataset are {0} rows and {1} columns.".format(gs[0], gs[1]))
    print()
    
    # Print datatypes of columns (features)
    games.info()
    print()
    
    # Descriptive statistics
    descriptive_stats(games)
    
    # Missing data analysis
    missing_data_analysis(games)
    
    # Fill empty elements with mean values
    games = games.fillna(games.mean())
    
    # Visualize distributions
    visualize_distributions(games)
    
    # Visualize categorical data
    visualize_categorical_data(games)
    
    # Detect outliers
    detect_outliers(games)
    
    # Separate train set and validation set
    X = games[["temp", "wspd", "pres", "dwpt"]]
    y = games[["SLG_PCT"]]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Dummy regressor model
    dummy_reg = DummyRegressor()
    dummy_reg.fit(X_train, y_train)
    y_pred = dummy_reg.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred, squared=False)
    print("DummyRegressor test error:", test_error)
    
    # Training error
    y_pred = dummy_reg.predict(X_train)
    training_error = mean_squared_error(y_train, y_pred, squared=False)
    print("DummyRegressor training error:", training_error)
    
    # LinearRegression model           
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Linear Regression test error:", rmse)
    
main()
