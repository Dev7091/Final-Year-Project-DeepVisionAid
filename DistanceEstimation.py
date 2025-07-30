import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading

# Constants
CONFIDENCE_THRESHOLD = 0.4

OBJECT_SIZES = {
    "person": 12,           # in inches
    "cell phone": 3.0,
    "car": 20.0,            # average width ~5.8 feet
    "motorbike": 15.0,       # average scooty width ~2.5 feet
    "dog": 10.0,          # average dog width ~1 foot
    "bottle": 2.0
}

LABEL_NAME_MAP = {
    "motorbike": "scooty"   # for speech
}

GREEN = (0, 255, 0)
RED = (0, 0, 255)
DARK_BLUE = (139, 0, 0)

FONTS = cv.FONT_HERSHEY_COMPLEX

model = YOLO('yolo11s.pt')

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

spoken_warnings = {}

def calibrate_focal_length(ref_image_paths, object_label, known_distance, known_width):
    focal_lengths = []
    for path in ref_image_paths:
        ref_image = cv.imread(path)
        if ref_image is None:
            print(f"Could not read image: {path}")
            continue
        detections = object_detector(ref_image)
        for d in detections:
            label, width_pixel, *_ = d
            if label == object_label and width_pixel > 0:
                focal_length = (width_pixel * known_distance) / known_width
                focal_lengths.append(focal_length)
    if not focal_lengths:
        raise ValueError(f"{object_label} not detected in any calibration images.")
    avg_focal_length = sum(focal_lengths) / len(focal_lengths)
    print(f"Average focal length for {object_label}: {avg_focal_length}")
    return avg_focal_length

def object_detector(image):
    results = model.predict(image, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    data_list = []

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = model.names[int(class_id)]
        width_pixel = x2 - x1
        height_pixel = y2 - y1
        box_area = width_pixel * height_pixel
        frame_area = image.shape[0] * image.shape[1]
        data_list.append([label, width_pixel, height_pixel, box_area, frame_area, (x1, y1, x2, y2)])
    return data_list

def calculate_distance(focal_length, real_width, width_pixel):
    if width_pixel == 0:
        return float('inf')  # Avoid division by zero
    return (real_width * focal_length) / width_pixel

def voice_warning(label):
    spoken_label = LABEL_NAME_MAP.get(label, label)
    if spoken_label not in spoken_warnings or not spoken_warnings[spoken_label]:
        spoken_warnings[spoken_label] = True
        threading.Thread(target=speak_warning, args=(spoken_label,)).start()

def speak_warning(label):
    message = f"Warning! The {label} is too close with less than a meter distance."
    engine.say(message)
    engine.runAndWait()

# Setup
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

reference_images = [
    "ReferenceImages/image1.png",
    "ReferenceImages/image2.png",
    "ReferenceImages/image3.png",
    "ReferenceImages/image4.png",
    "ReferenceImages/image5.png",
    "ReferenceImages/image6.png",
    "ReferenceImages/image7.png"
]

FOCAL_LENGTH = calibrate_focal_length(reference_images, "person", 45, OBJECT_SIZES["person"])

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    spoken_warnings.clear()
    detections = object_detector(frame)

    for d in detections:
        label, width_pixel, height_pixel, _, _, (x1, y1, x2, y2) = d

        if label in OBJECT_SIZES and width_pixel > 0:
            real_width = OBJECT_SIZES[label]
            distance_in_inches = calculate_distance(FOCAL_LENGTH, real_width, width_pixel)
            distance_in_meters = distance_in_inches * 0.0254

            # Draw bounding box and annotations
            cv.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv.putText(frame, label, (x1, y1 - 30), FONTS, 0.7, GREEN, 2)
            cv.putText(frame, f"{round(distance_in_inches, 2)} inches", (x1, y1 - 10), FONTS, 0.6, DARK_BLUE, 2)
            cv.putText(frame, f"{round(distance_in_meters, 2)} m", (x1, y1 + 20), FONTS, 0.6, RED, 2)

            if distance_in_meters < 1.0:
                warning_text = f"Warning! The {label} is too close with less than a meter distance."
                cv.putText(frame, warning_text, (50, 50), FONTS, 0.8, RED, 2)
                voice_warning(label)

    cv.imshow('Object Detection and Distance Measurement', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
