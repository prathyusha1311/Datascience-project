import websocket
import numpy as np
import pandas as pd
import orjson
import threading
import urllib.parse
import joblib
import time
from datetime import timedelta
import logging
import json
import requests
import os 
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_time_fall_detection.log"),  # Log to a file
        logging.StreamHandler()  # Also output logs to console
    ]
)

# ==========================
# Parameters and Settings
# ==========================

# File paths
MODEL_PATH = "lr_pre_6.0E+09solver_lbfgsiter_1000run_10.pkl"  # Update with your model path
LOG_FILE_PATH = "real_time_fall_detection.log"

# Sensor data settings
SLICE_SIZE = 6e9  # 6 seconds in nanoseconds
OVERLAP_SIZE = 1e9  # 1 second overlap in nanoseconds

# Sensitivity threshold for fall detection (adjust between 0 and 1)
THRESHOLD = 0.4  # Default threshold; lower value increases sensitivity

# WebSocket settings
WEBSOCKET_URL = "ws://Pixel-8.lan:8080/sensors/connect"  # Update with your WebSocket URL
SENSOR_TYPES = [
    "android.sensor.accelerometer",
    "android.sensor.gyroscope",
    "android.sensor.rotation_vector"
]

# User-specific settings (for future expansion)
USER_SETTINGS = {
    "default": {
        "threshold": THRESHOLD
    }
    # Add additional user configurations here
}

# ==========================
# Load the trained model
# ==========================
loaded_model = joblib.load(MODEL_PATH)

# ==========================
# Dummy Function for Fall Detection
# ==========================
from math import radians, sin, cos, sqrt, atan2


def calculate_distance(coord1, coord2):

    R = 3958.8  # Approximate radius of Earth in miles
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def send_hospital_telemetry_data(hospital_data):
    url = f"http://thingsboard.cloud/api/v1/XPPn7ty24eQg2Aq39Td8/telemetry"
    headers = {"Content-Type": "application/json"}
    
    # Format the data correctly for ThingsBoard
    telemetry_data = {
        "name": hospital_data["name"],
        "mylat": hospital_data["mylat"],
        "mylong": hospital_data["mylong"],
        "hospital_latitude": hospital_data["hospital_latitude"],
        "hospital_longitude": hospital_data["hospital_longitude"],
        "miles": hospital_data["miles"],
        "address": hospital_data["address"],
        "number": hospital_data["number"]
    }
    print(telemetry_data)
    response = requests.post(url, json=telemetry_data, headers=headers)
    if response.status_code == 200:
        print("Telemetry data sent successfully")
    else:
        print(f"Failed to send telemetry data. Status code: {response.status_code}")
        print(f"Response: {response.text}")

    return response

def find_closest_hospitals_osm(latitude, longitude, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    hospitals = []
    for i, element in enumerate(data['elements']):
        if i==3:
            break
        else:
            hospital = { 
                "name": element.get('tags', {}).get('name', 'Unknown'),
                "mylat" : latitude,
                "mylong" : longitude,
                "hospital_latitude": element.get('lat'),
                "hospital_longitude": element.get('lon'),
                "miles" : calculate_distance((latitude, longitude), (element.get('lat'),element.get('lon'))),
                "address": str(element.get('tags', {}).get('addr:housenumber', 'Unknown')) + ''+ str(element.get('tags', {}).get('addr:street', 'Unknown')) +''+
                        str(element.get('tags', {}).get('addr:postcode', 'Unknown')),
                "number": element.get('tags', {}).get('phone', 'Unknown')
            }
            hospitals.append(hospital)
    return hospitals  # Return the 3 nearest hospitals


# def find_closest_hospital_osm2(latitude, longitude, radius=5000):
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     overpass_query = f"""
#     [out:json];
#     (
#       node["amenity"="hospital"](around:{radius},{latitude},{longitude});
#       way["amenity"="hospital"](around:{radius},{latitude},{longitude});
#       relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
#     );
#     out center;
#     """

#     response = requests.get(overpass_url, params={'data': overpass_query})
#     data = response.json()
#     print(data)
#     if len(data['elements']) > 0:
#         closest = min(data['elements'], key=lambda x: (x.get('center', {}).get('lat', x.get('lat', latitude)) - latitude)**2 + (x.get('center', {}).get('lon', x.get('lon', longitude)) - longitude)**2)
#         return {
#             "name": closest.get('tags', {}).get('name', 'Unknown'),
#             "latitude": closest['center']['lat'],
#             "longitude": closest['center']['lon']
#         }
#     else:
#         return None
def send_hospital_telemetry_data(hospital_data):
      
    url = f"http://thingsboard.cloud/api/v1/XPPn7ty24eQg2Aq39Td8/telemetry"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=hospital_data, headers=headers)
    if response.status_code == 200:
        print("Telemetry data sent successfully")
    else:
        print(f"Failed to send telemetry data. Status code: {response.status_code}")
        print(f"Response: {response.text}")

    return response

def get_fall_location():
    def on_message(ws, message):
        data = json.loads(message)
        lat , long = data["latitude"] , data["longitude"]
        closest_hospitals = find_closest_hospitals_osm(lat, long)
        print("TTTTTT")
        print(closest_hospitals)
        for i in closest_hospitals:
            print(i)
            send_hospital_telemetry_data(i)
        ws.close()
    def on_error(ws, error):
        print("error occurred ", error)
    
    def on_close(ws, close_code, reason):
        print("connection closed : ", reason)
    
    def on_open(ws):
        print("connected")
        ws.send("getLastKnowLocation") # will calls back on_message when lastKnowLocation is not null
    

    def connect(url):
        ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

        ws.run_forever()
 
    connect("ws://Pixel-8.lan:8080/gps")
    sss = input("Are you Fine??? ")
    if sss == 'Yes':
        print("Cool")
    else:
        sys.exit()


def on_fall_detected():
    """Function to handle fall detection."""
    send_hospital_telemetry_data({'trigger':1})
    print("Fall detected! Taking appropriate action.")
    get_fall_location()
    # closest_hospital = find_closest_hospitals_osm(my_latitude, my_longitude)
    # if closest_hospital:
    #     print(f"Closest hospital: {closest_hospital['name']}")
    #     print(f"Coordinates: {closest_hospital['latitude']}, {closest_hospital['longitude']}")
    # else:
    #     print("No hospitals found nearby.")
    # Add code here to handle the fall event (e.g., send alert, log event)

# ==========================
# Helper Functions
# ==========================

def zero_crossings(slice):
    """Calculate zero crossings for a given slice."""
    return np.where(np.diff(np.signbit(slice)))[0].size

def min_max_distance(slice):
    """Calculate the Euclidean distance between min and max values."""
    min_val, max_val = np.amin(slice), np.amax(slice)
    return np.sqrt(np.square(max_val - min_val) + np.square(np.argmax(slice) - np.argmin(slice)))

def process_slice(slice_df):
    """Process a single slice of data to extract features."""
    slice_features = {}
    # Predefined feature names in the exact order used during training
    FEATURE_NAMES = [
        "gyro_x_min", "gyro_y_min", "gyro_z_min",
        "gyro_x_max", "gyro_y_max", "gyro_z_max",
        "x_min", "y_min", "z_min",
        "x_max", "y_max", "z_max",
        "x_std", "y_std", "z_std",
        "x_mean", "y_mean", "z_mean",
        "x_slope", "y_slope", "z_slope",
        "x_zc", "y_zc", "z_zc",
        "x_mmd", "y_mmd", "z_mmd",
        "pitch_slope", "roll_slope"
    ]

    # Gyroscope min and max
    gyro_channels = ['gyro_x', 'gyro_y', 'gyro_z']
    for channel in gyro_channels:
        data = slice_df[channel].to_numpy()
        slice_features[f'{channel}_min'] = np.amin(data)
        slice_features[f'{channel}_max'] = np.amax(data)

    # Accelerometer features
    acc_channels = ['x', 'y', 'z']
    for channel in acc_channels:
        data = slice_df[f'acc_{channel}'].to_numpy()
        slice_features[f'{channel}_min'] = np.amin(data)
        slice_features[f'{channel}_max'] = np.amax(data)
        slice_features[f'{channel}_std'] = np.std(data)
        slice_features[f'{channel}_mean'] = np.mean(data)
        slice_features[f'{channel}_slope'] = np.mean(np.diff(data))
        slice_features[f'{channel}_zc'] = zero_crossings(data)
        slice_features[f'{channel}_mmd'] = min_max_distance(data)

    # Orientation slopes
    for channel in ['pitch', 'roll']:
        data = slice_df[channel].to_numpy()
        slice_features[f'{channel}_slope'] = np.mean(np.diff(data))

    # Since we don't have labels in real-time data, set label to 0
    slice_features["label"] = 0

    return slice_features, FEATURE_NAMES

def run_one_LR(X_test, model, threshold=THRESHOLD):
    """Run the model on the test data with adjustable threshold."""
    probabilities = model.predict_proba(X_test)
    positive_class_probs = probabilities[:, 1]
    predictions = (positive_class_probs >= threshold).astype(int)
    return predictions

def calculate_orientation(values):
    """Convert rotation vector to azimuth, pitch, roll."""
    x, y, z, w = values[0], values[1], values[2], values[3] if len(values) > 3 else 1
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    azimuth = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)) * (180 / np.pi)
    pitch = np.arcsin(2 * (w * y - z * x)) * (180 / np.pi)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)) * (180 / np.pi)
    return azimuth, pitch, roll

def merge_sensor_data(acc_df, gyro_df, ori_df):
    """
    Merge accelerometer, gyroscope, and orientation data, aligning to gyroscope timestamps.
    """
    # Sort all DataFrames by timestamp without warnings
    acc_df = acc_df.sort_values("timestamp").reset_index(drop=True)
    gyro_df = gyro_df.sort_values("timestamp").reset_index(drop=True)
    ori_df = ori_df.sort_values("timestamp").reset_index(drop=True)

    # Define a tolerance for merging (e.g., 10 milliseconds in nanoseconds)
    tolerance = 10_000_000  # 10 milliseconds in nanoseconds

    # Merge accelerometer data to gyroscope data
    merged_df = pd.merge_asof(
        gyro_df, acc_df, on="timestamp",
        direction='nearest', tolerance=tolerance,
        suffixes=('', '_acc')
    )
    # Merge orientation data to the merged DataFrame
    merged_df = pd.merge_asof(
        merged_df, ori_df, on="timestamp",
        direction='nearest', tolerance=tolerance,
        suffixes=('', '_ori')
    )

    # Reorder columns for readability
    columns_order = [
        "timestamp", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "azimuth", "pitch", "roll"
    ]
    merged_df = merged_df[columns_order]

    # Drop rows with any NaN values resulting from unmatched merges
    merged_df.dropna(inplace=True)

    return merged_df

# Buffers for sensor data
acc_buffer = []
gyro_buffer = []
ori_buffer = []

# Initialize variables
buffer_start_time = None
slice_start_time = None
snum = 1  # Slice number

# ==========================
# WebSocket Handlers
# ==========================

def on_message(ws, message):
    """WebSocket message handler."""
    global buffer_start_time, slice_start_time, acc_buffer, gyro_buffer, ori_buffer, snum

    data = orjson.loads(message)
    timestamp_ns = data["timestamp"]
    sensor_type = data["type"]
    values = data["values"]

    # Collect data into buffers
    if buffer_start_time is None:
        buffer_start_time = timestamp_ns
        slice_start_time = timestamp_ns

    if sensor_type == "android.sensor.accelerometer":
        acc_buffer.append([timestamp_ns, *values])

    elif sensor_type == "android.sensor.gyroscope":
        gyro_buffer.append([timestamp_ns, *values])

    elif sensor_type == "android.sensor.rotation_vector":
        azimuth, pitch, roll = calculate_orientation(values)
        ori_buffer.append([timestamp_ns, azimuth, pitch, roll])
    # Check if we have enough data for a slice
    if timestamp_ns - slice_start_time >= SLICE_SIZE:
        # Process the data in the buffers up to the slice_start_time + SLICE_SIZE

        # Create DataFrames
        acc_df = pd.DataFrame(acc_buffer, columns=["timestamp", "acc_x", "acc_y", "acc_z"])
        gyro_df = pd.DataFrame(gyro_buffer, columns=["timestamp", "gyro_x", "gyro_y", "gyro_z"])
        ori_df = pd.DataFrame(ori_buffer, columns=["timestamp", "azimuth", "pitch", "roll"])
        # print("Acceletation:")
        # print(acc_df.head())
        # print("Gyro:")
        # print(gyro_df.head())
        # print("Ori:")
        # print(ori_df.head())

        # Filter data within the current slice
        slice_end_time = slice_start_time + SLICE_SIZE

        acc_slice = acc_df[(acc_df['timestamp'] >= slice_start_time) & (acc_df['timestamp'] < slice_end_time)]
        gyro_slice = gyro_df[(gyro_df['timestamp'] >= slice_start_time) & (gyro_df['timestamp'] < slice_end_time)]
        ori_slice = ori_df[(ori_df['timestamp'] >= slice_start_time) & (ori_df['timestamp'] < slice_end_time)]
        # Merge sensor data
        merged_df = merge_sensor_data(acc_slice, gyro_slice, ori_slice)

        if not merged_df.empty:
            # Process the merged DataFrame to extract features
            features, feature_order = process_slice(merged_df)

            # Prepare data for prediction
            X_test = [[features[feature] for feature in feature_order]]  # Use ordered features

            # Verify feature vector length
            expected_features = loaded_model.n_features_in_
            actual_features = len(X_test[0])
            if actual_features != expected_features:
                logging.error(f"Feature vector length {actual_features} does not match model's expected input {expected_features}")
            else:
                try:
                    # Run prediction with adjustable threshold
                    prediction = run_one_LR(X_test, loaded_model, threshold=THRESHOLD)

                    # Output prediction
                    prediction_label = "Fall Detected" if prediction[0] == 1 else "No Fall"
                    logging.info(f"Prediction for slice {snum}: {prediction_label}")
                    if prediction_label == 'Fall Detected':
                        send_hospital_telemetry_data({'trigger':1})
                    else:
                        send_hospital_telemetry_data({'trigger':0})

                    if prediction[0] == 1:
                        # Call the fall detected handler
                        on_fall_detected()

                except Exception as e:
                    logging.error(f"Error during prediction for slice {snum}: {e}")

        else:
            logging.warning(f"No data to process for slice {snum}")

        # Update slice_start_time for next slice
        # Use overlap to adjust the timing for real-time prediction improvement
        slice_start_time = slice_end_time - OVERLAP_SIZE
        snum += 1

        # Adjust buffer management to prevent data loss
        # Keep data beyond (slice_end_time - OVERLAP_SIZE) in buffers
        acc_buffer = [row for row in acc_buffer if row[0] >= (slice_end_time - OVERLAP_SIZE)]
        gyro_buffer = [row for row in gyro_buffer if row[0] >= (slice_end_time - OVERLAP_SIZE)]
        ori_buffer = [row for row in ori_buffer if row[0] >= (slice_end_time - OVERLAP_SIZE)]

def on_error(ws, error):
    """WebSocket error handler."""
    logging.error(f"WebSocket error occurred: {error}")

def on_close(ws, close_status_code, close_msg):
    """WebSocket close handler."""
    logging.info(f"WebSocket connection closed: {close_status_code}, {close_msg}")

def on_open(ws):
    """WebSocket open handler."""
    logging.info("Connected to WebSocket")

# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    #Build the WebSocket URL with parameters
    params = {'types': orjson.dumps(SENSOR_TYPES).decode("utf-8")}
    url = f"{WEBSOCKET_URL}?{urllib.parse.urlencode(params)}"

    # Initialize WebSocket app
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    # Run WebSocket
    ws.run_forever()