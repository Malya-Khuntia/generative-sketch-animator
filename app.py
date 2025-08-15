import os
import io
import time
import base64
import uuid
import logging
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

# Fetch Gemini API key (try .env first for local, then Secret Manager for Cloud Run)
# --- NEW: Securely Fetch API Key from Secret Manager ---
def get_gemini_api_key():
    """Fetches the Gemini API key from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        # The resource name of the secret version.
        name = f"projects/{PROJECT_ID}/secrets/gemini-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        # This provides a helpful error if the secret can't be fetched
        print(f"FATAL: Could not fetch secret 'gemini-api-key' from Secret Manager: {e}")
        print("Ensure the secret exists and the service account has 'Secret Manager Secret Accessor' role.")
        return None

logger.info("Fetching Gemini API key...")
API_KEY = get_gemini_api_key()
# Log partial key for debugging (first 5 and last 5 chars)
logger.info(f"API key retrieved: {API_KEY[:5]}...{API_KEY[-5:]}")

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
def upload_bytes_to_gcs(image_bytes: bytes, bucket_name: str, destination_blob_name: str) -> str:
    if not gcs_client:
        logger.error("GCS client is not initialized")
        raise ConnectionError("GCS client is not initialized")
    
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(image_bytes, content_type='image/png')
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        logger.info(f"Image uploaded to {gcs_uri}")
        return gcs_uri
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"GCS upload failed: {e}")
        raise

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video_from_sketch():
    if not all([gemini_image_client, veo_video_client, gcs_client]):
        logger.error("One or more clients not initialized")
        return jsonify({"error": "Server-side client initialization failed"}), 500

    if not request.json or 'image_data' not in request.json:
        logger.error("Missing image_data in request")
        return jsonify({"error": "Missing image_data in request"}), 400

    base64_image_data = request.json['image_data']
    user_prompt = request.json.get('prompt', '').strip()

    # --- Step 1: Generate Image with Gemini ---
    try:
        logger.info("Generating image from sketch with Gemini")
        if ',' in base64_image_data:
            base64_data = base64_image_data.split(',', 1)[1]
        else:
            base64_data = base64_image_data
        
        image_bytes = base64.b64decode(base64_data)
        sketch_pil_image = PIL.Image.open(io.BytesIO(image_bytes))

        default_prompt = "Convert this sketch into a photorealistic image as if it were taken from a real DSLR camera. The elements and objects should look real."
        response = gemini_image_client.models.generate_content(
            model=MODEL_ID_IMAGE,
            contents=[default_prompt, sketch_pil_image],
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
        )

        if not response.candidates:
            logger.error("Gemini returned no candidates")
            raise ValueError("Gemini image generation returned no candidates")

        generated_image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                generated_image_bytes = part.inline_data.data
                break
        
        if not generated_image_bytes:
            logger.error("Gemini response contained no image")
            raise ValueError("Gemini did not return an image")
        
        logger.info("Image generated successfully")

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API error: {e}")
        return jsonify({"error": f"Failed to generate image: {e}"}), 500
    except ValueError as e:
        logger.error(f"Image generation failed: {e}")
        return jsonify({"error": f"Failed to generate image: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in image generation: {e}")
        return jsonify({"error": f"Unexpected error in image generation: {e}"}), 500

    # --- Step 2 & 3: Upload Image to GCS and Generate Video with Veo ---
    try:
        logger.info("Uploading image to GCS")
        unique_id = uuid.uuid4()
        image_blob_name = f"images/generated-image-{unique_id}.png"
        output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/videos/"

        image_gcs_uri = upload_bytes_to_gcs(generated_image_bytes, GCS_BUCKET_NAME, image_blob_name)
        
        logger.info("Calling Veo to generate video")
        default_video_prompt = "Animate this image. Add subtle, cinematic motion."
        video_prompt = user_prompt if user_prompt else default_video_prompt

        operation = veo_video_client.models.generate_videos(
            model=MODEL_ID_VIDEO,
            prompt=video_prompt,
            image=types.Image(gcs_uri=image_gcs_uri, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                output_gcs_uri=output_gcs_prefix,
                duration_seconds=8,
                person_generation="allow_adult",
                enhance_prompt=True,
                generate_audio=True,
            ),
        )

        # TODO: For production, refactor to async (e.g., return job ID and poll via separate endpoint)
        timeout_seconds = 300
        start_time = time.time()
        while not operation.done:
            if time.time() - start_time > timeout_seconds:
                logger.error("Video generation timed out")
                raise TimeoutError("Video generation timed out")
            time.sleep(15)
            operation = veo_video_client.operations.get(operation)
            logger.info(f"Video generation status: {operation}")

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
    port = int(os.environ.get("PORT", 8080))  # Use PORT env var or default to 8080
    app.run(debug=True, host='0.0.0.0', port=port)
