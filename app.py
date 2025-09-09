import os
import uuid
import subprocess
import runpod
import boto3
import gc
import json
import sys
import logging
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, SeamlessM4Tv2Model

# --- Configuration ---
COMPUTE_TYPE = "float16"
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
S3_OUTPUT_DIR = "dubbingDemoOutput"
MAX_AUDIO_LENGTH = 30  # seconds

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

# Global variables for model and processor
model = None
processor = None
device = None

# ------------------- UTILITIES ------------------- #

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "max_cached": torch.cuda.max_memory_reserved() / 1024**3
        }
    return {}

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

def setup_model():
    """Initialize the model"""
    global model, processor, device
    
    try:
        # Clear memory first
        clear_gpu_memory()
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load model and processor
        logger.info("Loading model and processor...")
        model_name = "facebook/seamless-m4t-v2-large"
        
        processor = AutoProcessor.from_pretrained(
            model_name, 
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Load model with memory optimization
        model = SeamlessM4Tv2Model.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR
        )
        
        model.eval()
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV"""
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def speech_to_speech_dubbing(audio_path: str, target_lang: str):
    """Convert speech to speech in target language"""
    try:
        # Load and preprocess audio
        audio, orig_sr = sf.read(audio_path)
        
        # Resample if necessary
        if orig_sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Limit audio length to avoid memory issues
        max_samples = MAX_AUDIO_LENGTH * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.warning(f"Audio truncated to {MAX_AUDIO_LENGTH} seconds")
        
        audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000)
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
        
        # Generate output
        with torch.no_grad():
            audio_array = model.generate(**audio_inputs, tgt_lang=target_lang)[0].cpu()
        
        return audio_array
        
    except Exception as e:
        logger.error(f"Error in speech-to-speech: {str(e)}")
        raise

def text_to_speech_dubbing(text: str, target_lang: str):
    """Convert text to speech in target language"""
    try:
        # Process text
        text_inputs = processor(text=text, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Generate output
        with torch.no_grad():
            audio_array = model.generate(**text_inputs, tgt_lang=target_lang)[0].cpu()
        
        return audio_array
        
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        raise

def save_audio_to_s3(job_id, audio_path, language):
    """Save audio file to S3 bucket"""
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not configured, skipping audio save")
        return None
    
    try:
        directory_path = f"{S3_OUTPUT_DIR}/{job_id}/"
        file_key = f"{directory_path}dubbed_{language}.wav"
        
        s3.upload_file(audio_path, S3_BUCKET, file_key)
        
        s3_url = f"s3://{S3_BUCKET}/{file_key}"
        logger.info(f"Audio saved to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"Failed to save audio to S3: {str(e)}")
        return None

def handler(job):
    """RunPod serverless handler"""
    job_id = job.get("id", "unknown-job-id")
    
    try:
        # Check S3 configuration
        if not S3_BUCKET:
            return {"error": "S3_BUCKET environment variable not configured", "job_id": job_id, "status": "failed"}
        
        # Validate input
        if not job.get("input"):
            return {"error": "No input provided", "job_id": job_id, "status": "failed"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        text = input_data.get("text")
        target_language = input_data.get("target_language", "eng")
        
        # Check if we have text or audio input
        if not text and not file_name:
            return {"error": "Either text or file_name must be provided", "job_id": job_id, "status": "failed"}
        
        # Create temporary files
        temp_files = []
        
        try:
            if file_name:
                # 1. Download from S3
                local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
                try:
                    s3.download_file(S3_BUCKET, file_name, local_path)
                    temp_files.append(local_path)
                except Exception as e:
                    return {"error": f"S3 download failed: {str(e)}", "job_id": job_id, "status": "failed"}
                
                # 2. Convert to WAV
                try:
                    audio_path = convert_to_wav(local_path)
                    temp_files.append(audio_path)
                except Exception as e:
                    return {"error": f"Audio processing failed: {str(e)}", "job_id": job_id, "status": "failed"}

                # 3. Process audio
                result_audio = speech_to_speech_dubbing(audio_path, target_language)
                
                # Save output
                output_path = f"/tmp/{uuid.uuid4()}_dubbed.wav"
                sf.write(output_path, result_audio.numpy(), 16000)
                temp_files.append(output_path)
                
                # Upload to S3
                s3_url = save_audio_to_s3(job_id, output_path, target_language)
                
                response = {
                    "job_id": job_id,
                    "status": "success",
                    "output_path": output_path,
                    "s3_url": s3_url,
                    "target_language": target_language,
                    "gpu_memory": get_gpu_memory_usage()
                }
                
            else:
                # Process text
                result_audio = text_to_speech_dubbing(text, target_language)
                
                # Save output
                output_path = f"/tmp/{uuid.uuid4()}_tts.wav"
                sf.write(output_path, result_audio.numpy(), 16000)
                temp_files.append(output_path)
                
                # Upload to S3
                s3_url = save_audio_to_s3(job_id, output_path, target_language)
                
                response = {
                    "job_id": job_id,
                    "status": "success",
                    "output_path": output_path,
                    "s3_url": s3_url,
                    "target_language": target_language,
                    "gpu_memory": get_gpu_memory_usage()
                }
            
        except Exception as e:
            response = {"error": str(e), "job_id": job_id, "status": "failed"}

        finally:
            # Cleanup
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
            
            clear_gpu_memory()
            gc.collect()
        
        return response
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "job_id": job_id, "status": "failed"}

if __name__ == "__main__":
    print("🚀 Starting Dubbing API Endpoint...")
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Check environment variables
    if not S3_BUCKET:
        print("⚠️  WARNING: S3_BUCKET_NAME environment variable not set")
    
    # Setup model
    try:
        setup_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            sys.exit(1)
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        print("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("🔧 Running in local test mode...")
        # Test with mock input
        test_result = handler({
            "id": "test-job-id-123",
            "input": {
                "text": "Hello world, this is a test of the dubbing system.",
                "target_language": "spa"
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))