import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta


# Получение текущей погоды
def get_current_weather(api_key, location):
    base_url = "https://api.worldweatheronline.com/premium/v1/weather.ashx"
    params = {
        'key': api_key,
        'q': location,
        'fx': 'yes',
        'cc': 'yes',
        'format': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch data: {response.status_code}"}


# Получение исторической погоды за год
def get_historical_weather_year(api_key, location, year, end_date=None):
    start_date = f"{year}-01-01"
    end_date = end_date if end_date else f"{year}-12-31"
    base_url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    params = {
        'key': api_key,
        'q': location,
        'date': start_date,
        'enddate': end_date,
        'tp': 3,
        'format': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch data: {response.status_code}"}


# Наложение графиков для сравнения температур по годам
def plot_temperature_overlay(df):
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
    df['year'] = df['datetime'].dt.year
    df['normalized_datetime'] = df['datetime'].apply(lambda x: x.replace(year=2020))  # Фиктивный базовый год

    grouped = df.groupby('year')

    plt.figure(figsize=(12, 6))
    for year, group in grouped:
        plt.plot(group['normalized_datetime'], group['tempC'], label=str(year))

    plt.xlabel('Месяц и день')
    plt.ylabel('Температура (°C)')
    plt.xticks(rotation=90)
    plt.legend(title='Годы')
    plt.title('Сравнение температуры за разные годы')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Создание директории и сохранение файла
def save_file_in_directory(directory, filename, dataframe):
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_path = os.path.join(directory, f"{filename}.csv")
    excel_path = os.path.join(directory, f"{filename}.xlsx")
    dataframe.to_csv(csv_path, index=False)
    dataframe.to_excel(excel_path, index=False)


# Обучение модели и прогнозирование
def train_and_predict_weather(df):
    """
    Обучает модель на исторических данных и прогнозирует температуру на следующий временной интервал.
    """
    features = ['time', 'windspeedMiles', 'windspeedKmph', 'humidity', 'pressure', 'uvIndex']
    target = 'tempC'

    # Преобразование столбца time в числовой формат
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # Формирование обучающего и тестового наборов данных
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка точности модели
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Средняя абсолютная ошибка модели: {mae:.2f} °C")

    # Инициализация начальных признаков
    next_time_slot = X.iloc[0].copy()
    next_time_slot['time'] = (int(next_time_slot['time']) + 3) % 24  # Преобразуем в целое число и добавляем 3 часа

    # Создаем DataFrame с правильными признаками
    next_time_df = pd.DataFrame([next_time_slot], columns=features)

    next_temp = model.predict(next_time_df)[0]
    print(f"Прогноз температуры на следующий интервал: {next_temp:.2f} °C")

    # Прогноз на неделю вперед с использованием предыдущих данных
    predictions = []
    for i in range(7):
        # Обновляем время, добавляя 3 часа к каждому интервалу
        next_time_slot['time'] = (int(next_time_slot['time']) + 3) % 24

        # Обновление других признаков (в данном случае простое увеличение/уменьшение)
        next_time_slot['windspeedMiles'] = max(0, int(next_time_slot['windspeedMiles']) + (i - 3) % 2)
        next_time_slot['windspeedKmph'] = max(0, int(next_time_slot['windspeedKmph']) + (i - 3) % 2)
        next_time_slot['humidity'] = max(0, min(100, int(next_time_slot['humidity']) + (i - 3) % 3))
        next_time_slot['pressure'] = max(0, int(next_time_slot['pressure']) + (i - 2) % 2)

        next_date = datetime.now() + timedelta(days=i + 1)

        # Формирование DataFrame и получение прогноза для текущего дня
        next_time_df = pd.DataFrame([next_time_slot], columns=features)
        next_temp = model.predict(next_time_df)[0]
        predictions.append({'Date': next_date, 'Predicted Temp (°C)': round(next_temp, 2)})

    # Создаем DataFrame для визуализации прогноза
    prediction_df = pd.DataFrame(predictions)
    print("\nПрогноз на неделю вперед:")
    print(prediction_df)



# Отображение, обучение и прогнозирование данных
def display_train_and_predict_weather(api_key):
    location = input("Введите местоположение (например, London, UK): ")
    start_date_str = input("Введите начальную дату для исторической погоды (ГГГГ-ММ-ДД): ")
    end_date_str = input("Введите конечную дату (ГГГГ-ММ-ДД) или оставьте пустым: ")
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else datetime.now()
    location_dir = location.replace(' ', '_')
    current_time = datetime.now().strftime("%d%m%Y-%H%M")

    current_data = get_current_weather(api_key, location)
    print("\nТекущая погода:")
    if "current_condition" in current_data["data"]:
        current_weather = current_data["data"]["current_condition"][0]
        current_df = pd.DataFrame([current_weather])
        print(current_df)
        today_weather_filename = f"{location_dir}-today_weather-{current_time}"
        save_file_in_directory(location_dir, today_weather_filename, current_df)
    else:
        print("Текущая погода недоступна.")

    print("\nИсторическая погода:")
    all_days = []
    for year in range(start_date.year, end_date.year + 1):
        year_end_date = end_date.strftime("%Y-%m-%d") if year == end_date.year else f"{year}-12-31"
        historical_data = get_historical_weather_year(api_key, location, year, year_end_date)
        if "weather" in historical_data["data"]:
            for weather_day in historical_data["data"]["weather"]:
                for hour in weather_day['hourly']:
                    hour['date'] = weather_day['date']
                    all_days.append(hour)

    if all_days:
        historical_df = pd.DataFrame(all_days)
        print(historical_df)
        historical_weather_filename = f"{location_dir}-historical_weather-{current_time}"
        save_file_in_directory(location_dir, historical_weather_filename, historical_df)
        plot_temperature_overlay(historical_df)
        train_and_predict_weather(historical_df)
    else:
        print("Историческая погода недоступна.")


if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    display_train_and_predict_weather(api_key)
