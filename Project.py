# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define a class for handling S&P 500 market data
class SP500DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the S&P 500 market data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            # Rename 'Close/Last' column to 'Close' for consistency
            if 'Close/Last' in self.data.columns:
                self.data.rename(columns={'Close/Last': 'Close'}, inplace=True)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Failed to load data: {e}")

    def clean_data(self):
        """Clean the loaded S&P 500 market data."""
        if self.data is not None:
            self.data.dropna(inplace=True)  # Drop rows with any missing values
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])  # Convert date column to datetime format
            print("Data cleaning completed.")
        else:
            print("Data not loaded. Please load the data first.")

    def preprocess_data(self):
        """Preprocess the cleaned S&P 500 market data."""
        if self.data is not None and 'Close' in self.data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.data[['Close']] = scaler.fit_transform(self.data[['Close']])
            print("Data preprocessing completed.")
        else:
            if self.data is None:
                print("Data not loaded. Please load the data first.")
            else:
                print("Column 'Close' does not exist in the DataFrame. Please check the data.")

    def predictive_modeling(self):
        """Implement predictive modeling on the S&P 500 market data."""
        if self.data is not None:
            X = self.data[['Open', 'High', 'Low']]  # Adjusted features without 'Volume'
            y = self.data['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            print("Predictive modeling completed.")
        else:
            print("Data not preprocessed. Please preprocess the data first.")

    def get_clean_data(self):
        """Return the cleaned and preprocessed data."""
        if self.data is not None:
            return self.data
        else:
            print("Data is not ready. Please ensure data is loaded, cleaned, and preprocessed.")
            return None

# Example usage
if __name__ == "__main__":
    file_path = 'HistoricalData_1706833339339.csv'  # Update this path to your dataset
    data_handler = SP500DataHandler(file_path)
    data_handler.load_data()
    data_handler.clean_data()
    data_handler.preprocess_data()
    data_handler.predictive_modeling()
    clean_data = data_handler.get_clean_data()
    if clean_data is not None:
        print(clean_data.head())
    else:
        print("No data to display.")

plt.figure(figsize=(10, 6))  
plt.plot(clean_data['Date'], clean_data['Close'], label='Closing Price')  
plt.title('S&P 500 Closing Price Over Time')  
plt.xlabel('Date')  
plt.ylabel('Closing Price')  
plt.legend()  
plt.show()  


# Calculate moving averages  
clean_data['MA50'] = clean_data['Close'].rolling(window=50).mean()  
clean_data['MA200'] = clean_data['Close'].rolling(window=200).mean()  
  
plt.figure(figsize=(10, 6))  
plt.plot(clean_data['Date'], clean_data['Close'], label='Closing Price')  
plt.plot(clean_data['Date'], clean_data['MA50'], label='50-Day MA', color='orange')  
plt.plot(clean_data['Date'], clean_data['MA200'], label='200-Day MA', color='green')  
plt.title('S&P 500 Closing Price with Moving Averages')  
plt.xlabel('Date')  
plt.ylabel('Price')  
plt.legend()  
plt.show() 
  
# Assuming 'Close' is the column with closing prices  
df['Daily Returns'] = df['Close'].pct_change()  
df['Volatility'] = df['Daily Returns'].rolling(window=30).std() * np.sqrt(30)  
  
plt.figure(figsize=(10, 6))  
plt.plot(df['Date'], df['Volatility'], label='30-Day Rolling Volatility')  
plt.title('S&P 500 Volatility Over Time')  
plt.xlabel('Date')  
plt.ylabel('Volatility')  
plt.legend()  
plt.show() 