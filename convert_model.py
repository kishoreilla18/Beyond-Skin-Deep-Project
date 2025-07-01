from tensorflow.keras.applications import MobileNet

# Load MobileNet model (ImageNet pre-trained)
model = MobileNet(weights='imagenet', include_top=True)

# Save it in a new format
model.save("models/mobilenet_fixed.h5")

print("Model saved successfully!")
