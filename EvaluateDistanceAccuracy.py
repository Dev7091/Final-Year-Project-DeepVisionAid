import cv2
import numpy as np
import json
import os
import DistanceEstimation  # Import your existing DistanceEstimation.py

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load ground truth data
with open("ground_truth.json", "r") as f:
    ground_truth_data = json.load(f)  # Expected format: {"image1.png": {"person": actual_distance}}

# Storage for evaluation
true_distances = []
predicted_distances = []

# Evaluate all test images
for img_name, objects in ground_truth_data.items():
    img_path = os.path.join("TestImages", img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Image {img_name} not found!")
        continue

    # Run the distance estimation function from DistanceEstimation.py
    estimated_distances = DistanceEstimation.estimate_distance(frame)

    for obj_label, actual_distance in objects.items():
        if obj_label in estimated_distances:
            predicted_distance = estimated_distances[obj_label]
            true_distances.append(actual_distance)
            predicted_distances.append(predicted_distance)

# Compute distance estimation error
mae = mean_absolute_error(true_distances, predicted_distances) if true_distances else 0.0
rmse = mean_squared_error(true_distances, predicted_distances, squared=False) if true_distances else 0.0

# Print results
print(f"Distance Estimation Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f} inches")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} inches")
