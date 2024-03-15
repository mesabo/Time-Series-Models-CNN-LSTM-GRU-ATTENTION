import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

def load_and_create_dataset():
    # Generate synthetic dataset with 1000 rows and 4 columns
    num_rows = 1000
    num_columns = 4
    dataset = np.random.rand(num_rows, num_columns)
    # Convert to DataFrame
    dataset = pd.DataFrame(dataset, columns=['Feature1', 'Feature2', 'Feature3', 'Target'])
    return dataset

def normalize_and_split_dataset(dataset, look_back, forecast_days):
    # Separate features and target variable
    features = dataset.values
    target = dataset['Target'].values.reshape(-1, 1)

    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    # Normalize target variable
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target variable
    scaled_dataset = np.concatenate((scaled_features, scaled_target), axis=1)

    # Split into input and output
    X, y = [], []
    for i in range(len(scaled_dataset) - look_back - forecast_days + 1):
        X.append(scaled_dataset[i : (i + look_back), :-1])
        y.append(scaled_dataset[i + look_back : i + look_back + forecast_days, -1])

    X, y = np.array(X), np.array(y)

    # Split into train and test sets
    train_size = int(len(X) * 0.2)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler_features, scaler_target



def create_model(look_back, n_features, forecast_days):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(look_back, n_features)))
    # Output layer now predicts `forecast_days` values instead of 1
    model.add(Dense(units=forecast_days))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict_forecast(model, X_test):
    forecast = model.predict(X_test)
    return forecast

def calculate_l_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true) * 100


def main():
    # Step 1: Load and create dataset
    dataset = load_and_create_dataset()

    # Step 2: Set parameters
    look_back = 30
    forecast_days = 2
    epochs = 2
    batch_size = 64

    # Step 3: Normalize and split dataset
    X_train, X_test, y_train, y_test, scaler_features, scaler_target = normalize_and_split_dataset(
        dataset, look_back, forecast_days
    )

    # Step 4: Create model
    model = create_model(look_back, X_train.shape[2], forecast_days)

    # Step 5: Train model
    model = train_model(model, X_train, y_train, epochs, batch_size)

    # Step 6: Predict forecast days ahead
    forecast = predict_forecast(model, X_test)

    # Step 7: Inverse scaling for forecast
    forecast = scaler_target.inverse_transform(forecast)

    # Step 8: Inverse scaling for y_test
    y_test_inv = scaler_target.inverse_transform(y_test)

    # Step 9: Calculate L-RMSE
    l_rmse = calculate_l_rmse(y_test_inv, forecast)

    print("L-RMSE:", l_rmse)


if __name__ == "__main__":
    main()
