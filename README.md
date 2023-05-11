# Image Caption Generator

This is a Python script that generates captions for images using the Salesforce BLIP Image Captioning model and the OpenAI API. It provides a user-friendly interface using the Streamlit library.

## Prerequisites

Make sure you have the following dependencies installed:

- streamlit
- transformers
- openai
- tqdm
- PIL (Python Imaging Library)
- torch

You can install the dependencies by running the following command:

pip insntall -r requirements.txt

## Usage

To use the image caption generator, follow these steps:

1. Clone the repository and navigate to the project directory.
2. Run the script by executing the following command:
        streamlit run CapGen.py

1. The script will start a Streamlit server and open a web interface in your browser.
2. ![Screenshot (36)](https://github.com/AdityaRajPateriya/Image_caption_generator/assets/100833721/45fe3f0f-372b-4749-9a49-372bb0ed7fe0)

3. Click on the "Uload any Image" tab to upload your images.
   Select one or multiple images (in JPG, PNG, or JPEG format) using the file uploader.
4. Click the "Generate" button to generate captions for the uploaded images.
   The script will display the generated descriptions and captions for each image.
![Screenshot (37)](https://github.com/AdityaRajPateriya/Image_caption_generator/assets/100833721/5e823583-dbb9-4c20-8e0c-8912cc359b13)
![Screenshot (38)](https://github.com/AdityaRajPateriya/Image_caption_generator/assets/100833721/0696606d-a4f6-4b74-a4d4-8c56a4918ae6)


## How It Works
1. The script uses the Salesforce BLIP Image Captioning model to generate descriptions for the uploaded images. It then uses the bOpenAI API to generate creative captions based on the descriptions

2. The uploaded images are processed and displayed using the prediction function. This function opens each image, checks if it is in RGB format, and appends it to a list.
 
3. The pixel values of the images are extracted and converted to tensors.The model object generates captions for the pixel values using the generate method from the transformers library.
4. The generated captions are converted to text using the tokenizer object.The caption_generator function takes the generated        descriptions and uses the OpenAI API to generate creative captions for each description.
  It constructs a prompt for each description and sends it to the OpenAI API for completion.
  
5. The script displays the generated descriptions and captions using the Streamlit interface.
  Configuration
## Caution  
You need to provide your OpenAI API key and the model name for generating captions in the script. Modify the following lines in the code to add your credentials:
1. openai.api_key = "YOUR_OPENAI_API_KEY"
2. openai_model = "YOUR_OPENAI_MODEL_NAME"
3. I have used openai_model= "text-davinci-002"

## About the Author
This script was created by Aditya raj Pateriya. If you have any questions or suggestions, feel free to reach out.
Enjoy generating cool captions for your images!
  

