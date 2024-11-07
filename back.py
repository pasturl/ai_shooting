from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import replicate
import requests
from io import BytesIO
import zipfile
import base64
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    params: Dict[str, Any]
    model: str

class DownloadRequest(BaseModel):
    image_url: str
    prompt: str
    params: Dict[str, Any]

MODELS = {
    "flux-lora-tiger-wb-32-r-1-bz": "19b13186dae1abe145426ba7b85fd542d8a0691aecd758c82aae54d3715cfe92",
    "flux-lora-test-air-force-div-32-r-1-bz": "857513f74c939191ae8b7ba05510a13d228f56b9d753509117bbcc774087e243"
}

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image object"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    try:
        # Initialize Replicate client
        client = replicate.Client(api_token="YOUR_REPLICATE_API_TOKEN")  # Replace with env variable
        
        # Generate image
        output = client.run(
            f"{request.model}:{MODELS[request.model]}",
            input={
                "prompt": request.prompt,
                **request.params
            }
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            return {"image_url": output[0]}
        
        raise HTTPException(status_code=500, detail="No image generated")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download")
async def download_content(request: DownloadRequest):
    try:
        # Download the generated image
        image = download_image(request.image_url)
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Save the image
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            zip_file.writestr("generated_image.png", image_buffer.getvalue())
            
            # Save the prompt and parameters as text
            parameters_text = f"Prompt: {request.prompt}\nParameters:\n{request.params}"
            zip_file.writestr("parameters.txt", parameters_text)
        
        # Return the ZIP file
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=generated_content.zip"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)