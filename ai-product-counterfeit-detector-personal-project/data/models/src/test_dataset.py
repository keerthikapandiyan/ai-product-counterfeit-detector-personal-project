from dataset import load_dataset

images, labels = load_dataset("data/train")

print("Dataset loaded successfully!")
print("Number of images:", len(images))
print("Image shape:", images.shape)
print("Labels:", labels)