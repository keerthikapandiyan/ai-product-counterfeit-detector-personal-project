from dataset import load_dataset
from model import create_model
import os

# Load images and labels
images, labels = load_dataset("data/train")

print("Images:", images.shape)
print("Labels:", labels)

# Create CNN model
model = create_model()

# Train the model
model.fit(
    images,
    labels,
    epochs=10,
    batch_size=2,
    shuffle=True
)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
model.save("models/counterfeit_detector.keras")

print("Training completed!")
print("Model saved successfully!")