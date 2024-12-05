import streamlit as st
import replicate
import os
from PIL import Image
import requests
from io import BytesIO
import time
import zipfile
import base64
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import json
import asyncio
from datetime import datetime
import pathlib
import re
import concurrent.futures
from functools import partial
from typing import List, Dict
import random

# Configuration and styling
st.set_page_config(
    page_title="Flux Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Method 1: From environment variable (recommended for production)
# REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Method 2 Public: Direct input in app 

try:
    REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
except KeyError:
    REPLICATE_API_TOKEN = st.sidebar.text_input("Enter Replicate API Token", type="password")
    if not REPLICATE_API_TOKEN:
        st.warning("Please enter your Replicate API token to continue.")
        st.stop()

# Initialize replicate client with the provided token
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Replace the Anthropic API key initialization with:
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    ANTHROPIC_API_KEY = st.sidebar.text_input("Enter Anthropic API Key", type="password")
    if not ANTHROPIC_API_KEY:
        st.warning("Please enter your Anthropic API key to continue.")
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

def create_mock_variations(original_prompt: str, selected_model: str) -> Dict:
    """Generate mock variations for debug mode"""
    mock_variations = []
    styles = ["cinematic", "street photography", "fashion editorial", "urban lifestyle"]
    focuses = ["product detail", "full body shot", "environmental context", "artistic composition"]
    
    for i in range(10):
        mock_variations.append({
            "prompt": f"Mock variation {i+1} for: {original_prompt}",
            "style": random.choice(styles),
            "focus": random.choice(focuses)
        })
    
    return {"variations": mock_variations}

def create_mock_image() -> Image.Image:
    """Create a mock image for debug mode"""
    # Create a random colored image
    width, height = 512, 512
    mock_image = Image.new('RGB', (width, height))
    pixels = mock_image.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (random.randint(0, 255), 
                          random.randint(0, 255), 
                          random.randint(0, 255))
    return mock_image

def generate_prompt_variations(original_prompt, selected_model):
    """Generate 10 variations of the input prompt using Claude"""
    
    # Initialize Claude with better parameters for creative variations
    llm = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        model="claude-3-sonnet-20240229",
        temperature=0.8,
        max_tokens=2000
    )
    
    # Enhanced prompt template for better variations
    template = """You are an expert AI assistant specializing in generating professional-grade fashion photography prompts, with deep knowledge of commercial product photography, lighting techniques, and contemporary fashion campaigns.

CORE OBJECTIVES:
1. Generate 10 highly detailed, commercial-quality variations of the provided prompt
2. Ensure each prompt is optimized for AI image generation
3. Maintain consistent focus on the footwear product while creating compelling scene compositions
4. ALWAYS focus on the footwear product as the hero of the image.
5. NEVER include people on the images.

PROMPT REQUIREMENTS:
Each variation MUST include:

1. TECHNICAL SPECIFICATIONS
- Precise camera angles (e.g., "shot at 35mm, f/2.8 aperture")
- Specific lighting setup (e.g., "three-point lighting with rim light")
- Resolution and quality markers (e.g., "8K, hyperrealistic, photorealistic")
- Post-processing style (e.g., "slight film grain, Kodak Portra 400 colors")

2. ENVIRONMENT & COMPOSITION
- Detailed setting description
- Time of day and weather conditions
- Generate realistic scenes that match a footwear campaign.
- Specific composition rules (Rule of thirds, leading lines, etc.)
- Distance and framing (close-up, medium shot, wide shot)

4. PRODUCT EMPHASIS
- {selected_model} placement and interaction with environment
- Key product features to highlight
- Natural integration into the scene

5. ATMOSPHERE & MOOD
- Color palette and color grading
- Atmospheric elements (fog, shadows, reflections)
- Emotional tone of the image


FORBIDDEN ELEMENTS:
- Avoid generic descriptors (beautiful, nice, amazing)
- No unrealistic or physically impossible scenarios
- Avoid overshadowing the product with complex scenes
- No technical impossibilities for AI generation
    
    EXAMPLE PROMPT STRUCTURE:
    "Professional fashion campaign shot of {selected_model} sneakers, captured at 35mm with f/2.8 aperture. Three-point lighting setup with key light at 45 degrees. Setting: Modern concrete architecture with strong geometric shadows, shot during golden hour. Product positioned at lower third, emphasized by natural leading lines in architecture. Kodak Portra 400 color grading, slight film grain, 8K resolution. Style: minimal editorial fashion photography with strong architectural elements."

    Remember to make each variation unique while maintaining consistent professional quality and commercial viability. Focus on creating prompts that combine technical precision with artistic vision, always keeping the product as the hero of the image. 
    product: {selected_model}
    Original prompt: {original_prompt}
    
    Return ONLY a JSON response in this exact format:
    {{
        "variations": [
            {{
                "prompt": "complete prompt text",
                "style": "artistic style used",
                "focus": "main focus/perspective of this variation"
            }},
            // ... (9 more variations)
        ]
    }}
    
    Make each variation unique and creative while staying true to the original concept.
    Focus on creating high-quality, detailed prompts that will work well with AI image generation."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        with st.spinner("🤔 Generating prompt variations..."):
            # Get response from Claude
            response = llm.invoke(prompt.format(original_prompt=original_prompt, selected_model=selected_model))
            
            # Parse the JSON response
            try:
                # Extract the JSON part from the response
                json_str = response.content.strip()
                response_data = json.loads(json_str)
                
                # Create a container to display variations
                variation_container = st.container()
                
                with variation_container:
                    st.subheader("📝 Generated Prompt Variations")
                    cols = st.columns(2)
                    
                    for idx, var_data in enumerate(response_data.get("variations", []), 1):
                        col = cols[0] if idx <= 5 else cols[1]
                        with col:
                            with st.expander(f"Variation {idx}", expanded=False):
                                st.markdown(f"""
                                **Prompt:** {var_data['prompt']}
                                
                                **Style:** {var_data['style']}
                                
                                **Focus:** {var_data['focus']}
                                """)
                
                # Extract just the prompts for image generation
                variations = [item["prompt"] for item in response_data.get("variations", [])]
                
                # Store the full variation data in session state for later use
                st.session_state.variation_data = response_data.get("variations", [])
                
                return variations
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {str(e)}")
                return [original_prompt]

    except Exception as e:
        st.error(f"Error generating prompt variations: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return [original_prompt]  # Fallback to original prompt if generation fails

class FluxImageGenerator:
    def __init__(self):
        self.MODELS = {
            "Boy": {
                "Kids boy K099SS25 person product all 2k": {
                    "path": "pasturl/kids-boy-person-product-all-2k-20241204",
                    "version": "dd4d1f7de8688f3970701ca3845bf47333acf40d46c37c36c9caf73dd3127816"
                },
                "Kids boy K099SS25 person product all": {
                    "path": "pasturl/kids-boy-person-product-all-20241204",
                    "version": "10f3a469f79d8182293ca200a67abc39c5f1a0d28dba4e111f83c1f0c6001f38"
                },
                "Kids boy K099SS25 person selection": {
                    "path": "pasturl/kids-boy-person-all-selection-20241204",
                    "version": "442ba5c5135bff2516b5381df66c9e246369163c354f9b3b541920c6afb1b01f"
                },
                "Kids boy K099SS25 person all": {
                    "path": "pasturl/kids-boy-person-all-both-20241204",
                    "version": "bc7e3ea2bc02781dbafb270c7de6d0174288c2444b5dd060c0bc57116535ebb3"
                },
                "Kids boy K099SS25 product all": {
                    "path": "pasturl/kids-boy-all-both-20241204",
                    "version": "caeb33745d8279fc12ce866578ab6362ce427d800d94cc6d726258d2d92f7fc4"
                },
                "Kids boy K099SS25 all": {
                    "path": "pasturl/kids-boy-all-20241204",
                    "version": "d505363e63120456420e85c19ce88005cc6033d078bd69d7c8e58cca0a76714f"
                }
            },
            "Girl": {
                "Girl All K099SS25": {
                    "path": "pasturl/girl-all-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Girl Product K099SS25": {
                    "path": "pasturl/girl-product-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Girl Person K099SS25": {
                    "path": "pasturl/girl-person-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                }
            },
            "Man": {
                "Man Person K099SS25": {
                    "path": "pasturl/man-person-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Man Product K099SS25": {
                    "path": "pasturl/man-product-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Man All K099SS25": {
                    "path": "pasturl/man-all-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                }
            },
            "Woman": {
                "Woman All K099SS25": {
                    "path": "pasturl/woman-all-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Woman Person K099SS25": {
                    "path": "pasturl/woman-person-20241205",
                    "version": "VERSION_PLACEHOLDER"  # You'll provide this
                },
                "Woman Product K099SS25": {
                    "path": "pasturl/woman-product-20241205",
                    "version": "08746f00bc6f5551187429d5704852ebfeeff7c80c05e992dfa2ece659fdb0aa"  # You'll provide this
                }
            }
        }
        # Default to first category and model
        self.current_category = list(self.MODELS.keys())[0]
        self.current_model = list(self.MODELS[self.current_category].keys())[0]
    
    def set_model(self, category, model_name):
        """Set the current category and model"""
        self.current_category = category
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
        """Generate a single image"""
        try:
            model_info = self.MODELS[self.current_category][self.current_model]
            output = client.run(
                f"{model_info['path']}:{model_info['version']}",
                input={
                    "prompt": prompt,
                    **params
                },
                api_token=REPLICATE_API_TOKEN
            )
            
            if output and isinstance(output, list) and len(output) > 0:
                image = self.download_image(output[0])
                if image:
                    # Create timestamp for unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create base directory if it doesn't exist
                    base_dir = pathlib.Path("generated_images")
                    base_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create safe filename from prompt
                    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:30]
                    safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt).strip('-_')
                    
                    # Save image with timestamp and prompt in filename
                    filename = f"{timestamp}_{safe_prompt}.{params['output_format']}"
                    save_path = base_dir / filename
                    
                    # Save the image with the specified format and quality
                    image.save(
                        save_path, 
                        format=params['output_format'].upper(),
                        quality=params['output_quality']
                    )
                    
                    st.success(f"Image saved to: {save_path}")
                    return image
                
            return None
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None

    def generate_images_parallel(self, variations, params, max_workers=3):
        """Generate multiple images in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed params
            generate_func = partial(self.generate_image, params=params)
            
            # Submit all tasks and get future objects
            future_to_prompt = {
                executor.submit(generate_func, prompt): (i, prompt) 
                for i, prompt in enumerate(variations)
            }
            
            # Dictionary to store results
            results = {}
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_prompt):
                idx, prompt = future_to_prompt[future]
                try:
                    image = future.result()
                    results[idx] = {
                        'image': image,
                        'prompt': prompt,
                        'success': image is not None
                    }
                except Exception as e:
                    results[idx] = {
                        'image': None,
                        'prompt': prompt,
                        'success': False,
                        'error': str(e)
                    }
            
            # Sort results by original index
            return [results[i] for i in range(len(variations))]

def create_safe_filename(prompt, max_length=30):
    """Create a safe filename from the prompt"""
    # Remove special characters and replace spaces with underscores
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)
    safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt).strip('-_')
    # Truncate to max_length characters
    return safe_prompt[:max_length]

def save_generated_image(image, prompt, variation_num, params):
    """Save the generated image and its parameters in a timestamped folder"""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe filename from prompt
    safe_prompt = create_safe_filename(prompt)
    
    # Create base directory for saved images
    base_dir = pathlib.Path("generated_images")
    
    # Create folder with timestamp and prompt
    folder_name = f"{timestamp}_{safe_prompt}"
    save_dir = base_dir / folder_name
    
    # Create directories if they don't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_filename = f"variation_{variation_num}.png"
    image_path = save_dir / image_filename
    image.save(image_path, "PNG")
    
    # Save parameters
    params_filename = f"variation_{variation_num}_params.txt"
    params_path = save_dir / params_filename
    with open(params_path, "w") as f:
        f.write(f"Original Prompt: {prompt}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Variation: {variation_num}\n")
        f.write("\nParameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    return save_dir

def main():
    st.title("🎨 Flux Image Generator")
    
    # Initialize the generator
    generator = FluxImageGenerator()
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", help="Run with mock data instead of API calls")
    
    # Sidebar controls for model selection
    st.sidebar.title("Model Selection")
    
    # Category selector
    selected_category = st.sidebar.selectbox(
        "Choose Category",
        list(generator.MODELS.keys()),
        help="Select the model category"
    )
    
    # Model selector (filtered by category)
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        list(generator.MODELS[selected_category].keys()),
        help="Select which model to use for generation"
    )
    
    # Update the generator with selected model
    generator.set_model(selected_category, selected_model)
    
    st.sidebar.title("Generation Parameters")
    
    # Replace aspect_ratio with width and height selection
    format_options = {
        "Portrait (160x160)": {"width": 1440, "height": 1440},
        "Portrait Buchtreppe/Shoeshot (200x230)": {"width": 1252, "height": 1440},
        "Querformat (200x133)": {"width": 1440, "height": 957},
        "Promotion A (77x118.9)": {"width": 934, "height": 1440},
        "Langbahn Papier (75x180)": {"width": 600, "height": 1440},
        "Langbahn Stoff (99.5x230)": {"width": 623, "height": 1440},
        "Hochregalmotiv (1000x506)": {"width": 1440, "height": 728},
        "C-Format Quer Desktop": {"width": 1440, "height": 506},
        "C-Format Quer Mobile": {"width": 1440, "height": 960},
        "C-Format Hoch": {"width": 1116, "height": 1440}
    }

    selected_format = st.sidebar.selectbox(
        "Image Format",
        list(format_options.keys())
    )
    
    # Update params dictionary
    params = {
        "num_outputs": st.sidebar.selectbox("Number of Outputs", [1, 2, 3, 4], index=0),
        "width": format_options[selected_format]["width"],
        "height": format_options[selected_format]["height"],
        "model": st.sidebar.selectbox("Model", ["schnell", "dev"], index=0),
        "lora_scale": st.sidebar.slider("LoRA Scale", 0.0, 2.0, 1.0, 0.1),        
        "output_format": st.sidebar.selectbox("Output Format", ["png", "jpg", "webp"], index=0),
        "guidance_scale": st.sidebar.slider("Guidance Scale", 1.0, 20.0, 3.5, 0.5),
        "output_quality": st.sidebar.slider("Output Quality", 1, 100, 90, 1),
        "prompt_strength": st.sidebar.slider("Prompt Strength", 0.0, 1.0, 0.8, 0.1),
        "extra_lora_scale": st.sidebar.slider("Extra LoRA Scale", 0.0, 2.0, 1.0, 0.1),
        "num_inference_steps": st.sidebar.slider("Inference Steps", 1, 50, 4, 1)
    }
    
    # Main area
    prompt = st.text_area("Enter your prompt", height=100, 
                         help="Describe the image you want to generate")
    
    # Generation button
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt first.")
            return
            
        with st.spinner("🎨 Generating variations and images..."):
            # Generate prompt variations
            if debug_mode:
                mock_response = create_mock_variations(prompt, generator.current_model)
                variations = [var["prompt"] for var in mock_response["variations"]]
                st.session_state.variation_data = mock_response["variations"]
            else:
                variations = generate_prompt_variations(prompt, generator.current_model)
            
            if not variations:
                st.warning("Failed to generate prompt variations. Using original prompt only.")
                variations = [prompt]
            
            # Show progress message
            status = st.empty()
            status.text("🎨 Generating images in parallel...")
            
            # Generate images in parallel
            if debug_mode:
                results = [{'image': create_mock_image(), 
                          'prompt': var, 
                          'success': True} for var in variations]
            else:
                results = generator.generate_images_parallel(variations, params)
            
            # Create grid layout for thumbnails
            st.subheader("Generated Images")
            thumbnail_cols = st.columns(3)
            
            # First display all thumbnails
            for idx, result in enumerate(results):
                col = thumbnail_cols[idx % 3]
                with col:
                    if result['success']:
                        # Show thumbnail with click functionality
                        st.image(result['image'], 
                                caption=f"Variation {idx + 1}", 
                                use_column_width=True)
                        if st.button(f"Show Details #{idx + 1}"):
                            st.session_state.selected_image = idx
            
            # Show detailed view if an image is selected
            if hasattr(st.session_state, 'selected_image'):
                idx = st.session_state.selected_image
                result = results[idx]
                
                st.markdown("---")
                st.subheader(f"Variation {idx + 1} Details")
                
                # Create two columns for image and details
                img_col, details_col = st.columns([2, 1])
                
                with img_col:
                    st.image(result['image'], use_column_width=True)
                
                with details_col:
                    st.text_area("Prompt", value=result['prompt'], 
                               height=100, disabled=True)
                    
                    st.markdown("### Generation Parameters")
                    st.json(params)
                    
                    # Save image and create download button
                    save_dir = save_generated_image(result['image'], 
                                                  prompt, 
                                                  idx + 1, 
                                                  params)
                    
                    buffered = BytesIO()
                    with zipfile.ZipFile(buffered, "w") as zip_file:
                        image_buffer = BytesIO()
                        result['image'].save(image_buffer, format="PNG")
                        zip_file.writestr(
                            f"generated_image_variation_{idx + 1}.png", 
                            image_buffer.getvalue()
                        )
                        zip_file.writestr(
                            f"parameters_variation_{idx + 1}.txt",
                            f"Original Prompt: {prompt}\n"
                            f"Variation: {result['prompt']}\n"
                            f"Parameters:\n{params}"
                        )
                    
                    st.download_button(
                        label=f"Download Variation {idx + 1}",
                        data=buffered.getvalue(),
                        file_name=f"generated_content_variation_{idx + 1}.zip",
                        mime="application/zip"
                    )
                    
                    if st.button("Close Details"):
                        del st.session_state.selected_image
                        st.experimental_rerun()
            
            else:
                st.info("Click 'Show Details' under any image to see more information")
if __name__ == "__main__":
    main()
