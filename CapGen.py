import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import openai
from tqdm import tqdm
from PIL import Image
import torch

# object of processor,model,tokenizer
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Setting for the Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# openai api_Key and model name for generating multiple captions
openai.api_key = "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"
openai_model = "text-davinci-002"


# Defining method to generate caption
def caption_generator(des):
    caption_prompt = (''' Please generate three unique and creative captions to use on instagram for a photo that shows
    ''' + des + '''. The captions should be fun and creative.
    Captions:
    1.
    2.
    3.
    ''')

    # caption generation
    response = openai.Completion.create(
        engine=openai_model,
        prompt=caption_prompt,
        max_tokens=(175 * 3),
        n=1,
        stop=None,
        temperature=0.7
    )

    caption = response.choices[0].text.strip().split("\n")
    return caption


def prediction(img_list):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    img = []                                        # creating empty list to hold all images

    for image in tqdm(img_list):
        i_image = Image.open(image)                 # opening and storing image in i_image variable
        st.image(i_image, width=200)                # Displaying the uploaded image
        if i_image.mode != "RGB":                   # Checking if image in RGB, if not then convert into RGB
            i_image = i_image.convert(mode='RGB')

        img.append(i_image)                         # Appending the list in img[]

    # Extracting the pixel values
    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    # Generating output using pretrained model
    output = model.generate(pixel_val, **gen_kwargs)

    # Converting output of model to text
    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict


def upload():
    # from uploader inside tab
    with st.form("uploader"):
        # Image input
        image = st.file_uploader("upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        # generate button
        submit = st.form_submit_button("Generate")
        if submit:                                          # checking if the button is clicked
            description = prediction(image)

            st.subheader("description for the image:")
            for i, caption in enumerate(description):
                st.write(caption)
            st.subheader("captions for this image are:")
            captions = caption_generator(description[0])    # Function call to generate caption
            for caption in captions:
                st.write(caption)


def main():
    st.set_page_config(page_title="caption and hashtag generator")
    st.title("Cool Caption generator for your Images!!!")
    st.subheader('By Aditya raj Pateriya')

    tab1 = st.tabs("Uload any Image")
    upload()


if __name__ == '__main__':
    main()
