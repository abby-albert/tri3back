import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the attention dataset
attention_data = pd.read_csv('attention.csv')

# Drop NaN values
attention_data.dropna(inplace=True)

# Split the data into features and target
X = attention_data.drop(['subject', 'attention'], axis=1)  # Assuming 'subject' is irrelevant for prediction
y = attention_data['attention']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree regressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Test the model
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)