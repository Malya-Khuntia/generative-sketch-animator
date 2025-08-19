import os
import io
import time
import base64
import uuid
import logging
import datetime
import PIL.Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import types
from google.cloud import secretmanager

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load .env file for local testing
load_dotenv()

# Environment variables
MODEL_ID_IMAGE = 'gemini-2.0-flash-exp-image-generation'
MODEL_ID_VIDEO = 'veo-3.0-generate-preview'
PROJECT_ID = os.environ.get("PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

# Validate environment variables
REQUIRED_ENV_VARS = ["PROJECT_ID", "GCS_BUCKET_NAME", "GOOGLE_CLOUD_REGION"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

# --- Securely Fetch API Key from Secret Manager ---
def get_gemini_api_key():
    """Fetches the Gemini API key from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/gemini-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"FATAL: Could not fetch secret 'gemini-api-key' from Secret Manager: {e}")
        print("Ensure the secret exists and the service account has 'Secret Manager Secret Accessor' role.")
        return None

logger.info("Fetching Gemini API key...")
API_KEY = get_gemini_api_key()
if API_KEY:
    logger.info(f"API key retrieved: {API_KEY[:5]}...{API_KEY[-5:]}")
else:
    logger.error("API key could not be retrieved. The application may not function correctly.")

# Initialize clients
try:
    gemini_image_client = genai.Client(api_key=API_KEY)
    logger.info(f"Gemini Image Client initialized for model: {MODEL_ID_IMAGE}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    gemini_image_client = None

try:
    veo_video_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    logger.info(f"Veo Video Client initialized for project: {PROJECT_ID}")
except Exception as e:
    logger.error(f"Failed to initialize Veo client: {e}")
    veo_video_client = None

try:
    gcs_client = storage.Client(project=PROJECT_ID)
    logger.info("Google Cloud Storage Client initialized")
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {e}")
    gcs_client = None

# --- Helper Function to Upload to GCS ---
def upload_to_gcs(data: bytes or str, bucket_name: str, destination_blob_name: str, content_type: str) -> str:
    if not gcs_client:
        logger.error("GCS client is not initialized")
        raise ConnectionError("GCS client is not initialized")
    
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        if isinstance(data, str):
            blob.upload_from_string(data, content_type=content_type)
        else:  # bytes
            blob.upload_from_string(data, content_type=content_type)
            
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        public_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
        logger.info(f"Data uploaded to {gcs_uri}, public URL: {public_url}")
        return {"gcs_uri": gcs_uri, "public_url": public_url}
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"GCS upload failed: {e}")
        raise

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-images', methods=['POST'])
def generate_images():
    if not all([gemini_image_client, gcs_client]):
        logger.error("One or more clients not initialized")
        return jsonify({"error": "Server-side client initialization failed"}), 500

    if not request.json or 'image_data' not in request.json:
        logger.error("Missing image_data in request")
        return jsonify({"error": "Missing image_data in request"}), 400

    base64_image_data = request.json['image_data']
    user_prompt = request.json.get('prompt', '').strip()
    
    # Get and validate the number of images to generate from the request
    try:
        num_images = int(request.json.get('num_images', 4))
        if not 1 <= num_images <= 4:
            logger.warning(f"Invalid num_images value received: {num_images}. Defaulting to 4.")
            num_images = 4
    except (ValueError, TypeError):
        logger.warning("num_images was not a valid integer. Defaulting to 4.")
        num_images = 4

    # Create a unique folder for this generation job
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job_id = str(uuid.uuid4())
    job_folder_path = f"generations/{timestamp}_{job_id[:8]}"
    logger.info(f"Creating new job folder in GCS: {job_folder_path}")

    # Decode the image data once
    if ',' in base64_image_data:
        base64_data = base64_image_data.split(',', 1)[1]
    else:
        base64_data = base64_image_data
    image_bytes = base64.b64decode(base64_data)

    # Store original sketch and prompt in the new job folder
    try:
        logger.info(f"Uploading original user inputs for job {job_folder_path}")
        sketch_blob_name = f"{job_folder_path}/sketches/user-sketch.png"
        prompt_blob_name = f"{job_folder_path}/prompts/user-prompt.txt"
        
        upload_to_gcs(image_bytes, GCS_BUCKET_NAME, sketch_blob_name, 'image/png')
        upload_to_gcs(user_prompt or "No prompt provided.", GCS_BUCKET_NAME, prompt_blob_name, 'text/plain')
        
    except Exception as e:
        logger.error(f"Failed to upload original assets to GCS: {e}")
        pass

    # Generate the requested number of images with Gemini
    generated_images = []
    try:
        logger.info(f"Generating {num_images} images from sketch with Gemini")
        sketch_pil_image = PIL.Image.open(io.BytesIO(image_bytes))

        default_prompt = "Convert this sketch into a photorealistic image as if it were taken from a real DSLR camera. The elements and objects should look real."
        for i in range(num_images):
            logger.info(f"Generating image {i+1}")
            response = gemini_image_client.models.generate_content(
                model=MODEL_ID_IMAGE,
                contents=[default_prompt, sketch_pil_image],
                config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
            )

            if not response.candidates:
                logger.error(f"Gemini returned no candidates for image {i+1}")
                raise ValueError(f"Gemini image generation returned no candidates for image {i+1}")

            generated_image_bytes = None
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    generated_image_bytes = part.inline_data.data
                    break
            
            if not generated_image_bytes:
                logger.error(f"Gemini response contained no image for image {i+1}")
                raise ValueError(f"Gemini did not return an image for image {i+1}")
            
            # Upload generated image to GCS
            image_blob_name = f"{job_folder_path}/images/generated-image-{i+1}.png"
            upload_result = upload_to_gcs(generated_image_bytes, GCS_BUCKET_NAME, image_blob_name, 'image/png')
            generated_images.append({
                "public_url": upload_result["public_url"],
                "gcs_uri": upload_result["gcs_uri"]
            })
        
        logger.info("All images generated and uploaded successfully")
        return jsonify({"job_id": job_id, "images": generated_images})

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API error: {e}")
        return jsonify({"error": f"Failed to generate images: {e}"}), 500
    except ValueError as e:
        logger.error(f"Image generation failed: {e}")
        return jsonify({"error": f"Failed to generate images: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in image generation: {e}")
        return jsonify({"error": f"Unexpected error in image generation: {e}"}), 500

@app.route('/generate-video', methods=['POST'])
def generate_video():
    if not all([veo_video_client, gcs_client]):
        logger.error("One or more clients not initialized")
        return jsonify({"error": "Server-side client initialization failed"}), 500

    if not request.json or 'selected_image_gcs_uri' not in request.json or 'job_id' not in request.json:
        logger.error("Missing required fields in request")
        return jsonify({"error": "Missing selected_image_gcs_uri or job_id in request"}), 400

    selected_image_gcs_uri = request.json['selected_image_gcs_uri']
    user_prompt = request.json.get('prompt', '').strip()
    job_id = request.json['job_id']

    # Reconstruct job folder path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job_folder_path = f"generations/{timestamp}_{job_id[:8]}"

    try:
        logger.info(f"Generating video from selected image {selected_image_gcs_uri}")
        output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/{job_folder_path}/videos/"

        default_video_prompt = "Animate this image. Add subtle, cinematic motion."
        video_prompt = user_prompt if user_prompt else default_video_prompt

        operation = veo_video_client.models.generate_videos(
            model=MODEL_ID_VIDEO,
            prompt=video_prompt,
            image=types.Image(gcs_uri=selected_image_gcs_uri, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                output_gcs_uri=output_gcs_prefix,
                duration_seconds=8,
                person_generation="allow_adult",
                enhance_prompt=True,
                generate_audio=True,
            ),
        )

        timeout_seconds = 300
        start_time = time.time()
        while not operation.done:
            if time.time() - start_time > timeout_seconds:
                logger.error("Video generation timed out")
                raise TimeoutError("Video generation timed out")
            time.sleep(15)
            operation = veo_video_client.operations.get(operation)
            logger.info(f"Video generation status: {operation.metadata.state.name if operation.metadata else 'pending'}")

        logger.info("Video generation operation complete")
        
        if not operation.response or not operation.result.generated_videos:
            logger.error("Veo returned no video")
            raise ValueError("Veo operation completed but returned no video")

        video_gcs_uri = operation.result.generated_videos[0].video.uri
        logger.info(f"Video saved to GCS at: {video_gcs_uri}")
        
        video_blob_name = video_gcs_uri.replace(f"gs://{GCS_BUCKET_NAME}/", "")
        public_video_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{video_blob_name}"
        logger.info(f"Video public URL: {public_video_url}")

        return jsonify({"generated_video_url": public_video_url})

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Veo or GCS API error: {e}")
        return jsonify({"error": f"Failed to generate video: {e}"}), 500
    except TimeoutError as e:
        logger.error(f"Video generation timeout: {e}")
        return jsonify({"error": f"Failed to generate video: {e}"}), 504
    except ValueError as e:
        logger.error(f"Video generation failed: {e}")
        return jsonify({"error": f"Failed to generate video: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in video generation: {e}")
        return jsonify({"error": f"Unexpected error in video generation: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
