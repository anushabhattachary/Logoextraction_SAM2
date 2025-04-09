import os
import torch
from torchvision import transforms
from extract_frames import extract_frames
from segment_logos_sam import load_sam, segment_and_save
from logo_classifier import get_logo_classifier
from predict_teams import predict_logo
from save_predictions import save_predictions_csv

# === CONFIG ===
video_path = "sample_match.mp4"
frames_dir = "data/frames"
logos_dir = "data/logos"
predictions_csv = "outputs/predictions.csv"
checkpoint_path = "best_model.pth"  # Make sure to place your model here

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(logos_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# === 1. Extract Frames ===
print("🎞️ Extracting frames from video...")
extract_frames(video_path, frames_dir, frame_rate=1)

# === 2. Segment Logos with SAM ===
print("🔍 Segmenting logos using SAM...")
predictor = load_sam()
for frame_file in sorted(os.listdir(frames_dir)):
    frame_path = os.path.join(frames_dir, frame_file)
    output_path = os.path.join(logos_dir, frame_file)
    segment_and_save(predictor, frame_path, output_path)

# === 3. Load Model ===
print("🧠 Loading trained logo classifier...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_logo_classifier(num_classes=10).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class_names = ['Barcelona', 'RealMadrid', 'ManUnited', 'Bayern', 'PSG', 
               'Chelsea', 'Juventus', 'Liverpool', 'ManCity', 'ACMilan']

# === 4. Predict Teams ===
print("📊 Predicting teams from logos...")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

predictions = []
for logo_file in sorted(os.listdir(logos_dir)):
    logo_path = os.path.join(logos_dir, logo_file)
    team, confidence = predict_logo(logo_path, model, class_names, transform, device)
    predictions.append({
        "frame": logo_file,
        "team": team,
        "confidence": confidence
    })
    print(f"{logo_file}: {team} ({confidence:.2f})")

# === 5. Save Predictions ===
save_predictions_csv(predictions, predictions_csv)
print("✅ Done! Predictions saved.")