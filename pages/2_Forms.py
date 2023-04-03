import openai
import streamlit as st
import pytesseract
from PIL import Image
import os
import pandas as pd
import csv

openai.api_key = "<INSERT API KEY HERE>"
model_engine = "text-davinci-002"

labels = {
    "Name": ["full name", "first name", "last name"],
    "Gender": ["gender", "sex"],
    "Date Of Birth": ["date of birth", "dob"],
    "Email": ["email", "e-mail"],
    "Phone Number": ["phone number", "telephone number", "mobile number"],
    "Address": ["address", "location"],
    "College": ["college", "university"]
    }

# Function to extract text from uploaded images using Pytesseract
def extract_text(images):
    texts = []
    for image in images:
        text = pytesseract.image_to_string(Image.open(image), lang='eng')
        texts.append(text)
    return texts

# Function to filter data using OpenAI's GPT-3
def filter_data():
    filtered_data_list = []
    num_images = 3  # Change the number of images here

    with open('filtered_forms.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Filtered Data'])

        for i in range(num_images):
            # Load the extracted text from the invoice image
            with open(f"image{i+1}.txt", 'r') as file:
                extracted_text = file.read()

            # Generate the prompt dynamically based on the available labels
            prompt = "Label-Fields Extraction for Document Management\n\nForm field labels:\n"
            for label in labels:
                prompt += "- " + label + "\n"
            prompt += "\nExtracted text:\n" + extracted_text + "\n\nFiltered data:\n"

            # Generate the filtered data using GPT-3
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.5,
            )

            filtered_data = response.choices[0].text.strip()
            filtered_data_list.append(filtered_data)

            # Write the filtered data to the CSV file
            writer.writerow([f"image{i+1}", filtered_data])

    # Display the filtered data for each image
    for i, filtered_data in enumerate(filtered_data_list):
        st.write(f"Image {i + 1}:")
        st.write(filtered_data)

    st.write("Filtered data saved to filtered_data.csv")
    #if filtered_data_csv has null values, then fill it with "No Data"
    df = pd.read_csv('filtered_forms.csv')
    df.fillna('No Data', inplace=True)
    df.to_csv('filtered_invoices.csv', index=False)


st.set_page_config(page_title="Object Detection for Invoices")


# Create a sidebar with options
option = st.sidebar.selectbox(
    'Select an option',
    ('Home', 'Extract Data', 'Filter Data'))

# Show the appropriate page based on the user's option
if option == 'Home':
    st.title("Object Detection for Invoices")
    st.write("Welcome to the Object Detection for Invoices app!")

elif option == 'Extract Data':
    st.subheader("Extract Data")
    uploaded_files = st.file_uploader("Choose one or more images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if uploaded_files:
        # Display uploaded images and extracted text for each image
        texts = extract_text(uploaded_files)
        for i, (uploaded_file, text) in enumerate(zip(uploaded_files, texts)):
            st.write(f"Image {i + 1}:")
            st.image(uploaded_file, use_column_width=True)
            st.write(f"Extracted text {i + 1}:")
            
            # Display extracted text in a monospace font
            st.markdown(f"```{text}```")
            
            # Save extracted text to a txt file
            filename = f"image{i + 1}.txt"
            with open(filename, "w") as f:
                f.write(text)
        st.write(f"Extracted text saved to {filename}")

elif option == 'Filter Data':
    st.subheader("Filter Data")
    filter_data()
