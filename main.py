from transformers import pipeline
from PIL import Image
import gradio as gr



classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify_image(image):
    results = classifier(image)
    output = ""
    for result in results:
        label = result['label']
        confidence = result['score'] * 100
        output += f"{label}: {confidence:.1f}% confident\n"

    return output

# print("\n"*20)

interface = gr.Interface(
    fn=classify_image, # The funciton it calls
    inputs=gr.Image(type="pil"), # What goes in as input
    outputs="text", # What comes out as output
    title="Image Classifier", # The title of the tab
    description="Upload an image and it will get automatically classified", # The description
    examples=[], # No examples as of now
)

if __name__ == "__main__":
    interface.launch()