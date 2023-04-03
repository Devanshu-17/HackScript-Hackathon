import pytesseract
from PIL import Image
import os
import csv
import openai

openai.api_key = "<INSERT API KEY HERE>"
model_engine = "text-davinci-002"

# Define the available labels
labels = {
    "Seller": ["seller", "vendor"],
    "Buyer Address": ["buyer address", "billing address"],
    "Price": ["price", "total"],
    "Date": ["date", "invoice date"],
    "Invoice No": ["invoice no", "number"],
    "Product": ["product", "item"],
    "Quantity": ["quantity", "qty"],
    "Unit Price": ["unit price", "price"]
}

# Create a list to store the extracted text for each image
extracted_text_list = []

# Loop through all images in the directory
for filename in os.listdir('dataset'):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load image and extract text using Pytesseract
        image_path = os.path.join('dataset', filename)
        text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
        
        # Add the extracted text and image filename to the list
        extracted_text_list.append({'Filename': filename, 'Text': text})

# Save the extracted text to a text file with headings for each image
with open('extracted_text.txt', 'w') as file:
    for extracted_text in extracted_text_list:
        file.write(extracted_text['Filename'] + '\n\n' + extracted_text['Text'] + '\n\n')

# Generate the prompt dynamically based on the available labels
prompt = "Label-Fields Extraction for Document Management\n\nForm field labels:\n"
for label in labels:
    prompt += "- " + label + "\n"
prompt += "\n"

# Create a list to store the filtered data for each image
filtered_data_list = []

# Loop through the extracted text for each image and filter the data using GPT-3
for extracted_text in extracted_text_list:
    # Generate the prompt with the extracted text for the current image
    prompt += "Extracted text for " + extracted_text['Filename'] + ":\n" + extracted_text['Text'] + "\n\nFiltered data:\n"
    
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    filtered_data = response.choices[0].text.strip()
    filtered_data_list.append({'Filename': extracted_text['Filename'], 'Filtered Data': filtered_data})

# Save the filtered data to a CSV file
with open('filtered_data.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Filename', 'Filtered Data'])
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Filtered Data'])
    for filtered_data in filtered_data_list:
        writer.writerow([filtered_data['Filename'], filtered_data['Filtered Data']])

# Print the filtered data for each image
for filtered_data in filtered_data_list:
    print(filtered_data['Filename'] + '\n\n' + filtered_data['Filtered Data'] + '\n\n')

