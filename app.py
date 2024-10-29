import streamlit as st
import replicate
import os
from PIL import Image
import requests
from io import BytesIO
import time
import zipfile
import base64
# Configuration and styling
st.set_page_config(
    page_title="Flux Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Method 1: From environment variable (recommended for production)
# REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Method 2 Public: Direct input in app 

REPLICATE_API_TOKEN = st.sidebar.text_input("Enter Replicate API Token", type="password")
if not REPLICATE_API_TOKEN:
    st.warning("Please enter your Replicate API token to continue.")
    st.stop()


# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .stProgress>div>div>div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Function to trigger automatic download of ZIP file
def trigger_download(zip_data, filename="generated_content.zip"):
    b64 = base64.b64encode(zip_data).decode()
    download_script = f"""
        <a href="data:application/zip;base64,{b64}" download="{filename}">Click here if the download does not start automatically</a>
        <script>
            var link = document.createElement("a");
            link.href = "data:application/zip;base64,{b64}";
            link.download = "{filename}";
            link.click();
        </script>
    """
    st.markdown(download_script, unsafe_allow_html=True)

class FluxImageGenerator:
    def __init__(self):
        self.MODELS = {
            "Tiger Model SNKTGWB": {
                "path": "pasturl/flux-lora-tiger-wb-32-r-1-bz",
                "version": "19b13186dae1abe145426ba7b85fd542d8a0691aecd758c82aae54d3715cfe92"
            },
            "Air Force Model LORAAIRFORCE": {
                "path": "pasturl/flux-lora-test-air-force-div-32-r-1-bz",
                "version": "857513f74c939191ae8b7ba05510a13d228f56b9d753509117bbcc774087e243" 
            }
        }
        # Default to first model
        self.current_model = list(self.MODELS.keys())[0]
        
    def set_model(self, model_name):
        """Set the current model"""
        self.current_model = model_name
        
    def download_image(self, url):
        """Download image from URL and return PIL Image object"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            st.error(f"Error downloading image: {str(e)}")
            return None
        
    def generate_image(self, prompt, params):
        try:
            model_info = self.MODELS[self.current_model]
            output = replicate.run(
                f"{model_info['path']}:{model_info['version']}",
                input={
                    "prompt": prompt,
                    **params
                }
            )
            if output and isinstance(output, list) and len(output) > 0:
                return self.download_image(output[0])
            return None
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None

def main():
    st.title("🎨 Flux Image Generator")
    
    # Initialize the generator
    generator = FluxImageGenerator()
    
    # Sidebar controls
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        list(generator.MODELS.keys()),
        help="Select which Flux model to use for generation"
    )
    generator.set_model(selected_model)
    st.sidebar.title("Generation Parameters")
    
    # Model parameters
    params = {
        "num_outputs": st.sidebar.selectbox("Number of Outputs", [1, 2, 3, 4], index=0),
        "aspect_ratio": st.sidebar.selectbox(
            "Aspect Ratio",
            ["1:1", "4:5", "3:4", "2:3", "16:9", "9:16", "3:2", "5:7"],
            index=0
        ),
        "model": st.sidebar.selectbox("Model", ["dev", "schnell "], index=0),
        "lora_scale": st.sidebar.slider("LoRA Scale", 0.0, 2.0, 1.0, 0.1),        
        "output_format": st.sidebar.selectbox("Output Format", ["png", "jpg", "webp"], index=0),
        "guidance_scale": st.sidebar.slider("Guidance Scale", 1.0, 20.0, 3.5, 0.5),
        "output_quality": st.sidebar.slider("Output Quality", 1, 100, 90, 1),
        "prompt_strength": st.sidebar.slider("Prompt Strength", 0.0, 1.0, 0.8, 0.1),
        "extra_lora_scale": st.sidebar.slider("Extra LoRA Scale", 0.0, 2.0, 1.0, 0.1),
        "num_inference_steps": st.sidebar.slider("Inference Steps", 1, 50, 28, 1)
    }
    
    # Main area
    prompt = st.text_area("Enter your prompt", height=100, 
                         help="Describe the image you want to generate")
    
    # Generation button
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt first.")
            return
            
        with st.spinner("🎨 Generating your image..."):
            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Generate image
            generated_image = generator.generate_image(prompt, params)
            
            if generated_image:
                # Display generation parameters
                with st.expander("Generation Details", expanded=False):
                    st.json(params)
                
                # Display the generated image
                st.success("✨ Image generated successfully!")
                st.image(generated_image, caption="Generated Image", use_column_width=True)
                
                # Create a download button for the image
                buffered = BytesIO()
                with zipfile.ZipFile(buffered, "w") as zip_file:
                    # Save the image
                    image_buffer = BytesIO()
                    generated_image.save(image_buffer, format="PNG")
                    zip_file.writestr("generated_image.png", image_buffer.getvalue())
                    
                    # Save the prompt and parameters as text
                    zip_file.writestr("parameters.txt", f"Prompt: {prompt}\nParameters:\n{params}")
                # Trigger the automatic download
                trigger_download(buffered.getvalue())

                st.download_button(
                    label="Download ZIP",
                    data=buffered.getvalue(),
                    file_name="generated_content.zip",
                    mime="application/zip"
                )
            else:
                st.error("Failed to generate image. Please try again.")
if __name__ == "__main__":
    main()