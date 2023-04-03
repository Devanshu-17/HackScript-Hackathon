from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification

# Load the LayoutLMv3 model and processor
processor = AutoProcessor.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")
model = AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")



# Load the image
image_path = "1.png"
image = Image.open(image_path)

# If the image only has 2 dimensions, add a third dimension for the channel
if len(image.size) == 2:
    image = image.convert("RGB")

# Resize the image
resized_image = image.resize((224, 224))

# Preprocess the image
inputs = processor(image, return_tensors="pt", padding=True)

# Run the model on the image
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)

# Get the bounding boxes and labels from the predictions
tokens = processor.batch_decode(predictions)[0]
boxes = processor.decode_boxes(outputs.prediction_boxes[0], inputs.data[0]["attention_mask"])
labels = [processor.id2label[label_id] for label_id in predictions[0]]

# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(image)
for i in range(len(tokens)):
    box = boxes[i]
    label = labels[i]
    draw.rectangle(box, outline="red")
    draw.text((box[0], box[1]), label)

# Display the image
image.show()
