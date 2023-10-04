import pandas as pd
import psycopg2
import os
import sys
import numpy as np
import psycopg2.extras

# Define a function to remove non-numeric characters (except ".") from a string

CHUNK_SIZE = 3000


def remove_non_numeric(text):
    new_value = ''.join(filter(lambda x: x.isdigit() or x == '.', text))

    return new_value if new_value != '' else np.nan

def remove_non_characters(text):
    # Using a generator expression and join to filter out non-character elements
    return ''.join(char for char in text if char.isalpha() or (char == " " or char.isdigit()))

def preprocess(df):
        # Make a copy of the input DataFrame
        df = df.copy()

        print(df.head(3))

        # Combine 'Datum (Anlage)' and 'Zeit (Anlage)' columns into a single 'DateTime' column and drop the original columns
        df['DateTime'] = pd.to_datetime(df['Datum (Anlage)'] + ' ' + df['Zeit (Anlage)'],
                                        format='%d.%m.%y %H:%M:%S')
        df.drop(columns=['Datum (Anlage)', 'Zeit (Anlage)'], inplace=True)

        # Reducing dimention for the Dataset
        df = df[::3]

        df.set_index('DateTime', inplace=True)

        # Define a list of columns with missing data exceeding 25%
        missing_values_columns = ['Ereignis', 'Error Number', 'Wind Speed (max)', 'Wind Speed (min)',
                                  'Rotor Speed [rpm] (max)', 'Rotor Speed [rpm] (min)',
                                  'Active Power (max)', 'Active Power (min)', 'Wind Direction (avg)',
                                  'Feature 0', 'Feature 5', 'Feature 6', 'Feature 8', 'Feature 9',
                                  'Feature 10', 'Feature 11', 'Feature 13', 'Feature 14', 'Feature 15',
                                  'Feature 19', 'Feature 20', 'Feature 21', 'Feature 22', 'Feature 23',
                                  'Feature 24', 'Feature 25', 'Feature 26', 'Feature 27', 'Feature 29',
                                  'Feature 30', 'Feature 31', 'Feature 32', 'Feature 33', 'Feature 34',
                                  'Feature 35', 'Feature 36', 'Feature 37', 'Feature 38', 'Feature 39',
                                  'Feature 40', 'Feature 41', 'Feature 42', 'Feature 43', 'Feature 44',
                                  'Feature 45', 'Feature 46', 'Feature 47', 'Feature 48', 'Feature 49',
                                  'Feature 50', 'Feature 51', 'Feature 52', 'Feature 53', 'Feature 54',
                                  'Feature 55', 'Feature 56', 'Feature 57', 'Feature 58', 'Feature 59',
                                  'Feature 60', 'Feature 61', 'Feature 62', 'Feature 63', 'Feature 64',
                                  'Feature 65', 'Feature 66', 'Feature 67', 'Feature 68', 'Feature 69',
                                  'Feature 70', 'Feature 71', 'Feature 72', 'Feature 73', 'Feature 74',
                                  'Feature 75', 'Feature 76', 'Feature 77', 'Feature 78', 'Feature 79',
                                  'Feature 80', 'Feature 81', 'Feature 82']

        # Drop columns with missing data exceeding 25%
        df = df.drop(missing_values_columns, axis=1)

        # Replace commas with dots in all columns
        df = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)

        # Remove non-numeric characters from all columns
        df = df.applymap(lambda x: remove_non_numeric(x) if isinstance(x, str) else x)

        # Convert all columns (except the last one) to float
        df.iloc[:, :] = df.iloc[:, :].astype(float)

        # Drop columns with high correlation
        df.drop(columns=["Generator Speed [rpm] (avg)",
                         "Feature 16", "Feature 17",
                         "Feature 2", "Feature 18", "Feature 4",
                         "Reactive Power (avg)",
                         ], axis=1, inplace=True)

        # Drop rows with missing data in the 'Wind Speed (avg)' column
        df.dropna(subset=["Wind Speed (avg)"], inplace=True)

        # Forward-fill missing data
        df.fillna(method="ffill", inplace=True)

        df = df.sort_values(by='DateTime', ascending=True)

        hour = 60 * 60
        minute = 60
        day = 60 * 60 * 24
        year = 365.2425 * day

        df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

        df['Day sin'] = np.sin(df['Seconds'] * (2 * np.pi / day))
        df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
        df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
        df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
        df['hour sin'] = np.sin(df['Seconds'] * (2 * np.pi / hour))
        df['hour cos'] = np.cos(df['Seconds'] * (2 * np.pi / hour))
        df['minute sin'] = np.sin(df['Seconds'] * (2 * np.pi / minute))
        df['minute cos'] = np.cos(df['Seconds'] * (2 * np.pi / minute))
        df = df.drop('Seconds', axis=1)

        df.columns = [remove_non_characters(i) for i in df.columns]

        return df

def save_to_database(df, cursor):
    # Prepare data tuples and insert query
    data_tuples = [(
        row['Wind Speed avg'],
        row['Rotor Speed rpm avg'],
        row['Active Power avg'],
        row['Nacelle Position avg'],
        row['Feature 1'],
        row['Feature 3'],
        row['Feature 7'],
        row['Feature 28'],
        row['Day sin'],
        row['Day cos'],
        row['Year sin'],
        row['Year cos'],
        row['hour sin'],
        row['hour cos'],
        row['minute sin'],
        row['minute cos']
    ) for _, row in df.iterrows()]

    insert_query = """
        INSERT INTO "TurbineData" (
            "WindSpeedAvg", "RotorSpeedRpmAvg", "ActivePowerAvg", "NacellePositionAvg", 
            "Feature1", "Feature3", "Feature7", "Feature28", "DaySin", "DayCos", 
            "YearSin", "YearCos", "HourSin", "HourCos", "MinuteSin", "MinuteCos"
        ) VALUES %s
        ON CONFLICT (id) DO NOTHING
    """

    psycopg2.extras.execute_values(cursor, insert_query, data_tuples)

def process_uploaded_file(file_path):
    connection = psycopg2.connect(
        dbname="mydb",
        user="myuser",
        password="mypassword",
        host="localhost",
        port="5432"
    )
    cursor = connection.cursor()

    for chunk in pd.read_csv(file_path, on_bad_lines='skip', low_memory=False, delimiter=";",skiprows=3, index_col=0, chunksize=CHUNK_SIZE):
        df = preprocess(chunk)
        save_to_database(df, cursor)

    connection.commit()
    cursor.close()
    connection.close()


if __name__ == "__main__":
    uploaded_file_path = sys.argv[1]
    process_uploaded_file(uploaded_file_path)
