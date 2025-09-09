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
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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

def list_files_with_size(directory: str):
    """List files in a directory with size in MB"""
    files_info = []
    try:
        for root, _, files in os.walk(directory):
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    files_info.append({
                        "path": fpath,
                        "size_mb": round(size_mb, 2)
                    })
                except Exception:
                    pass
    except Exception as e:
        files_info.append({"error": str(e)})
    return files_info

def get_system_usage():
    """Return disk, memory, and file listings"""
    usage = {}
    try:
        # Disk usage
        disk = subprocess.check_output(["df", "-h", "/"]).decode("utf-8").split("\n")[1].split()
        usage["disk_total"] = disk[1]
        usage["disk_used"] = disk[2]
        usage["disk_available"] = disk[3]
        usage["disk_percent"] = disk[4]

        # Memory usage
        mem_output = subprocess.check_output(["free", "-h"]).decode("utf-8").split("\n")
        if len(mem_output) > 1:
            mem_parts = mem_output[1].split()
            usage["mem_total"] = mem_parts[1]
            usage["mem_used"] = mem_parts[2]
            usage["mem_free"] = mem_parts[3]
            usage["mem_shared"] = mem_parts[4]
            usage["mem_cache"] = mem_parts[5]
            usage["mem_available"] = mem_parts[6]

        # Files in tmp + cache
        usage["tmp_files"] = list_files_with_size("/tmp")
        usage["model_cache_files"] = list_files_with_size(MODEL_CACHE_DIR)

        # Summarize downloaded models
        usage["models_summary"] = []
        if os.path.exists(MODEL_CACHE_DIR):
            for subdir in os.listdir(MODEL_CACHE_DIR):
                model_path = os.path.join(MODEL_CACHE_DIR, subdir)
                if os.path.isdir(model_path):
                    total_size_mb = 0
                    for root, _, files in os.walk(model_path):
                        for f in files:
                            try:
                                total_size_mb += os.path.getsize(os.path.join(root, f))
                            except:
                                pass
                    usage["models_summary"].append({
                        "model_name": subdir,
                        "size_mb": round(total_size_mb / (1024 * 1024), 2)
                    })

    except Exception as e:
        usage["error"] = str(e)

    # Add GPU memory info
    usage["gpu_memory"] = get_gpu_memory_usage()
    return usage

# ------------------- MAIN HANDLER ------------------- #

def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def setup_model():
    """Initialize the model from pre-downloaded cache"""
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
        
        # Check if model is already downloaded
        model_local_path = os.path.join(MODEL_CACHE_DIR, "seamless-m4t-v2-large")
        if not os.path.exists(model_local_path):
            logger.warning("Model not found in cache, downloading now...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="facebook/seamless-m4t-v2-large",
                local_dir=model_local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        
        # Load model and processor from local cache
        logger.info("Loading model and processor from cache...")
        
        processor = AutoProcessor.from_pretrained(
            model_local_path,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Load model with memory optimization
        model = SeamlessM4Tv2Model.from_pretrained(
            model_local_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR
        )
        
        model.eval()
        logger.info("✅ Model loaded successfully from cache!")
        
        # Verify model is working
        with torch.no_grad():
            test_input = processor(text="Hello", return_tensors="pt")
            if device == "cuda":
                test_input = {k: v.to(device) for k, v in test_input.items()}
            test_output = model.generate(**test_input, tgt_lang="eng")
            logger.info("✅ Model test passed!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback: try loading without device_map
        try:
            logger.info("Trying fallback loading without device_map...")
            model_local_path = os.path.join(MODEL_CACHE_DIR, "seamless-m4t-v2-large")
            model = SeamlessM4Tv2Model.from_pretrained(
                model_local_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir=MODEL_CACHE_DIR
            )
            model = model.to(device)
            model.eval()
            logger.info("✅ Model loaded successfully with fallback method!")
        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed: {str(fallback_error)}")
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

def save_response_to_s3(job_id, response_data, status="success"):
    """Save response to S3 bucket"""
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not configured, skipping response save")
        return False
    
    try:
        directory_path = f"{S3_OUTPUT_DIR}/{job_id}/"
        file_key = f"{directory_path}response.json"
        
        response_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=response_json,
            ContentType='application/json'
        )
        
        logger.info(f"Response saved to S3: s3://{S3_BUCKET}/{file_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save response to S3: {str(e)}")
        return False

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
            response = {"error": "S3_BUCKET environment variable not configured", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response
        
        # Validate input
        if not job.get("input"):
            response = {"error": "No input provided", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        text = input_data.get("text")
        target_language = input_data.get("target_language", "eng")
        
        # Check if we have text or audio input
        if not text and not file_name:
            response = {"error": "Either text or file_name must be provided", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response
        
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
                    response = {"error": f"S3 download failed: {str(e)}", "job_id": job_id, "status": "failed"}
                    save_response_to_s3(job_id, response, "failed")
                    return response
                
                # 2. Convert to WAV
                try:
                    audio_path = convert_to_wav(local_path)
                    temp_files.append(audio_path)
                except Exception as e:
                    response = {"error": f"Audio processing failed: {str(e)}", "job_id": job_id, "status": "failed"}
                    save_response_to_s3(job_id, response, "failed")
                    return response

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
            
            # Save successful response
            save_response_to_s3(job_id, response, "success")
            
        except Exception as e:
            response = {"error": str(e), "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")

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
        
        response["system_usage"] = get_system_usage()
        return response
        
    except Exception as e:
        response = {"error": f"Unexpected error: {str(e)}", "job_id": job_id, "status": "failed"}
        save_response_to_s3(job_id, response, "failed")
        return response

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
    
    # Verify model cache directory
    if not ensure_model_cache_dir():
        print("❌ ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            sys.exit(1)
    
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