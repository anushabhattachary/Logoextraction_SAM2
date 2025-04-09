# outputs/save_predictions.py

import csv
import json
import os

def save_predictions_csv(predictions, filepath="outputs/predictions.csv"):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "predicted_team", "confidence"])
        for pred in predictions:
            writer.writerow([pred["frame"], pred["team"], f"{pred['confidence']:.2f}"])
    print(f"✅ Predictions saved to {filepath}")

def save_predictions_json(predictions, filepath="outputs/predictions.json"):
    with open(filepath, "w") as file:
        json.dump(predictions, file, indent=2)
    print(f"✅ Predictions saved to {filepath}")
