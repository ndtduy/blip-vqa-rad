import os
import uuid
import io
import pandas as pd
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import streamlit as st

def inference(model, processor, image=None, image_path=None, question_text=None):
    if image is None and image_path is not None:
        image = Image.open(image_path).convert('RGB')
    inputs = processor(image, question_text, padding=True, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

def inferenceOnUploadedImage(model, processor):
    uploaded_file = st.file_uploader('Choose an radio image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        if not os.path.exists('upload_images'):
            os.makedirs('upload_images')
        uploaded_file_path = os.path.join('upload_images', uuid.uuid4().hex + '_' + uploaded_file.name)
        uploaded_file_bytes = uploaded_file.read()
        with open(uploaded_file_path, 'wb') as f:
            f.write(uploaded_file_bytes)
        st.image(uploaded_file_bytes, caption='Uploaded Image', use_column_width=True)
    else:
        # st.warning('Please upload an image!')
        pass
    
    question_text = st.text_input('Ask a question about the radiology image!')
    if st.button('Ask'):
        with st.spinner('Inference is running...'):
            answer = inference(model, processor, image_path=uploaded_file_path, question_text=question_text)
        st.success(f'Answer: {answer}')

def inferenceOnSampleImage(model, processor):
    # vqa-rad dataset, get test set
    splits = {'train': 'data/train-00000-of-00001-eb8844602202be60.parquet', 'test': 'data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'}
    samples = pd.read_parquet('hf://datasets/flaviagiammarino/vqa-rad/' + splits['test'])
    sample = samples.sample(1, replace=False)

    with st.container():
        sample_image_bytes = io.BytesIO(sample['image'].item()['bytes'])
        image = Image.open(sample_image_bytes).convert('RGB')
        question_text = sample['question'].item()

        st.image(sample_image_bytes, caption='Sample Image', use_column_width=True)
        st.info(f'Sample Question: {question_text}')
        st.info(f'Actual Answer: {sample["answer"].item()}')
        with st.spinner('Inference is running...'):
            answer = inference(model, processor, image=image, question_text=question_text)
        st.success(f'Predicted Answer: {answer}')
    
    random_sample_button = st.button('Random a Sample') # Unused but necessary to trigger the event
    
def runTask(choice, model, processor):
    if choice == 'Upload an image':
        inferenceOnUploadedImage(model, processor)
    elif choice == 'Use a sample image':
        inferenceOnSampleImage(model, processor)

def main():
    st.set_page_config(page_title='Visual Question Answering on Radiology Image', page_icon=':hugging_face:')
    st.title('Visual QA tailored on Radiology Image')
    st.caption('Hugging Face\'s Transformers, PyTorch, Streamlit')

    with st.sidebar:
        st.title('Visual QA tailored on Radiology Image')
        
        choices = st.selectbox(
            'Choose an option',
            ('Upload an image', 'Use a sample image'),
            index=0
        )
        '[![](https://img.shields.io/badge/GitHub%20Repo-F9AB00?style=for-the-badge&logo=github&labelColor=grey&color=black)](https://github.com/ndtduy/blip-vqa-rad)'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
    model = BlipForQuestionAnswering.from_pretrained('hop1um/blip-vqa-rad').to(device)
    runTask(choices, model, processor)
if __name__ == '__main__':
    main()
