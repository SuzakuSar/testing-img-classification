from transformers import pipeline
from PIL import Image

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

results = classifier("Black.webp")


print("\n"*20)
for result in results:
    label = result['label']
    confidence = result['score'] * 100
    print(f"{label}: {confidence:.1f}% confident\n")