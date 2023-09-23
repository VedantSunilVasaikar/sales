from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import uuid
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
from urllib.parse import quote
from urllib.parse import unquote

app = Flask(__name__)
auth = HTTPBasicAuth()

# Define a username and password for authentication
USERS = {
    "username": "password"  # Replace with your desired credentials
}

# Create a temporary directory to store uploaded datasets
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function to sanitize filenames
def sanitize_filename(filename):
    # Replace invalid characters with underscores
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Remove spaces and extra underscores
    filename = filename.replace(' ', '_').replace('__', '_').strip('_')
    return filename

# Define a function for SARIMA forecasting
def perform_forecasting(data, start_date, end_date):
    try:
        # Convert 'Order Date' to a datetime column with error handling
        data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')

        # Drop rows with invalid dates (if any)
        data = data.dropna(subset=['Order Date'])

        # Define the SARIMA model parameters
        p, d, q = 1, 1, 1  # Autoregressive, differencing, and moving average orders
        P, D, Q, s = 1, 1, 1, 12  # Seasonal autoregressive, seasonal differencing, seasonal moving average, and seasonal period

        # Get unique categories
        categories = data['Category'].unique()

        # List to store the forecast results
        forecast_results = []

        # Loop through each category and fit a SARIMA model
        for category in categories:
            # Filter data for the current category
            category_data = data[data['Category'] == category]

            # Group the data by 'Order Date' and aggregate monthly sales
            category_sales = category_data.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()

            # Define the training and prediction period
            train_data = category_sales[category_sales['Order Date'] < start_date]
            prediction_period = pd.date_range(start=start_date, end=end_date, freq='M')

            # Fit the SARIMA model to the current category
            model = SARIMAX(train_data['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, s))
            results = model.fit()

            # Forecast future values
            forecast_steps = len(prediction_period)
            forecast = results.get_forecast(steps=forecast_steps)

            # Get the forecasted values and corresponding dates
            forecasted_values = forecast.predicted_mean
            forecasted_dates_numeric = forecast.row_labels  # Numeric date positions

            # Convert numeric dates to human-readable format (e.g., "YYYY-MM")
            forecasted_dates = [start_date + pd.DateOffset(months=i) for i in range(len(forecasted_dates_numeric))]

            # Create a dictionary for the current category's forecast
            category_forecast = {
                'Category': category,
                'Forecasted_Sales': forecasted_values.tolist(),
                'Forecasted_Dates': [date.strftime("%Y-%m") for date in forecasted_dates],  # Format dates as "YYYY-MM"
            }

            # Generate a plot of training data and forecasts
            plt.figure(figsize=(10, 6))
            plt.plot(train_data['Order Date'], train_data['Sales'], label='Training Data', marker='o')
            plt.plot(forecasted_dates, forecasted_values, label='Forecast', linestyle='--', marker='o')
            plt.title(f'Category: {category}')
            plt.xlabel('Order Date')
            plt.ylabel('Sales')
            plt.legend()

            # Save the plot as an image file
            plot_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'plot_{sanitize_filename(category)}.png')
            plt.savefig(plot_filename, format='png')
            plt.close()

            # Add the plot filename to the category_forecast dictionary
            category_forecast['Plot_Filename'] = plot_filename

            # Append the category's forecast to the results list
            forecast_results.append(category_forecast)

        return forecast_results

    except Exception as e:
        return str(e)

# Route to upload a dataset and perform forecasting
@auth.login_required
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            # Read the uploaded dataset
            data = pd.read_csv(file)

            # Get start_date and end_date from the JSON request
            request_data = request.form
            start_date = pd.to_datetime(request_data['start_date'])
            end_date = pd.to_datetime(request_data['end_date'])

            # Call the forecasting function with the uploaded dataset
            forecast_results = perform_forecasting(data, start_date, end_date)

            # Create a response with forecast results and plot URLs
            response_data = {'message': 'Forecasting completed successfully', 'forecast_results': forecast_results}

            # Return the response with JSON data
            return jsonify(response_data), 200

        else:
            return jsonify({'error': 'Invalid file format'})

    except Exception as e:
        return jsonify({'error': str(e)})


# Securely serve uploaded files with authentication
@auth.login_required
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)




