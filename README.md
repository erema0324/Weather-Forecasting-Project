# Weather Forecasting Project

## Description

This project provides tools for obtaining and analyzing weather data, as well as predicting temperature using historical data and machine learning. It leverages the World Weather Online API to fetch current and historical weather data, and includes functions for visualization and forecasting.

## Features

- Fetch current weather for a specified location.
- Retrieve historical weather data for a selected year.
- Visualize temperature data with overlay plots for comparing different years.
- Save weather data in CSV and Excel formats.
- Train a machine learning model to predict temperature.
- Forecast temperature for the upcoming week.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_repository/weather-forecasting-project.git
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Obtain an API key from [World Weather Online](https://www.worldweatheronline.com/) and replace `"YOUR_API_KEY"` in the `Weather.py` file with your API key.

## Usage

Run the script:

    ```bash
    python Weather.py
    ```

Follow the on-screen instructions to input the location and date range. The script will automatically fetch current weather data, historical data, and create a forecast for the upcoming week.

## Example Usage

Enter location (e.g., London, UK):  
Enter start date for historical weather (YYYY-MM-DD):  
Enter end date (YYYY-MM-DD) or leave empty:

The script will create and save weather data files in CSV and Excel formats, visualize the data, and output a temperature forecast.


