import streamlit as st
from main import ImagePredictor, ImprovedTinyVGG
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define your custom image transform function
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    # Add any other transformations you want to apply
])

# Load your saved model
MODEL_SAVE_PATH = "models/04_pytorch_custom_datasets_model_2.pth"  
class_names = ["apple_pie", "chocolate_mousse", "waffles", "red_velvet_cake", "cheesecake"]  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

model = ImprovedTinyVGG(input_shape=3, hidden_units=3, output_shape=len(class_names))

model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
model.to(device)

# Initialize the ImagePredictor
image_predictor = ImagePredictor(model=model, class_names=class_names, device=device)

# Define the Streamlit app
def main():
    st.title("Image Classifier")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            predicted_label = image_predictor.pred_and_plot_image(uploaded_image=image, custom_image_transform=custom_image_transform)
            st.write(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()



