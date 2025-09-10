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

# Logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

# Global model vars (lazy loaded)
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
    """Lazy-load the model"""
    global model, processor, device
    if model is not None and processor is not None:
        return  # already loaded

    try:
        clear_gpu_memory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        model_name = "facebook/seamless-m4t-v2-large"

        processor = AutoProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
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
        logger.error(f"Model loading failed: {str(e)}")
        # Add more specific error information
        if "CUDA" in str(e):
            logger.error("CUDA might not be properly configured")
        raise e

def convert_to_wav(input_path: str) -> str:
    """Convert to 16kHz mono wav"""
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", "-loglevel", "error", output_path
    ], check=True)
    return output_path

def speech_to_speech_dubbing(audio_path: str, target_lang: str):
    setup_model()
    audio, orig_sr = sf.read(audio_path)

    if orig_sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    max_samples = MAX_AUDIO_LENGTH * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        logger.warning(f"Audio truncated to {MAX_AUDIO_LENGTH} seconds")

    audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000)
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

    with torch.no_grad():
        audio_array = model.generate(**audio_inputs, tgt_lang=target_lang)[0].cpu()
    return audio_array

def text_to_speech_dubbing(text: str, target_lang: str):
    setup_model()
    text_inputs = processor(text=text, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        audio_array = model.generate(**text_inputs, tgt_lang=target_lang)[0].cpu()
    return audio_array

def save_audio_to_s3(job_id, audio_path, language):
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not configured")
        return None
    directory_path = f"{S3_OUTPUT_DIR}/{job_id}/"
    file_key = f"{directory_path}dubbed_{language}.wav"
    s3.upload_file(audio_path, S3_BUCKET, file_key)
    return f"s3://{S3_BUCKET}/{file_key}"

def handler(job):
    job_id = job.get("id", "unknown-job-id")
    try:
        if not job.get("input"):
            return {"error": "No input provided", "job_id": job_id, "status": "failed"}

        input_data = job["input"]
        file_name = input_data.get("file_name")
        text = input_data.get("text")
        target_language = input_data.get("target_language", "eng")

        temp_files, response = [], {}
        if file_name:
            # Download from S3
            local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
            s3.download_file(S3_BUCKET, file_name, local_path)
            temp_files.append(local_path)

            audio_path = convert_to_wav(local_path)
            temp_files.append(audio_path)

            result_audio = speech_to_speech_dubbing(audio_path, target_language)
            output_path = f"/tmp/{uuid.uuid4()}_dubbed.wav"
            sf.write(output_path, result_audio.numpy(), 16000)
            temp_files.append(output_path)

            s3_url = save_audio_to_s3(job_id, output_path, target_language)
            response = {"job_id": job_id, "status": "success", "s3_url": s3_url,
                        "target_language": target_language, "gpu_memory": get_gpu_memory_usage()}
        else:
            result_audio = text_to_speech_dubbing(text, target_language)
            output_path = f"/tmp/{uuid.uuid4()}_tts.wav"
            sf.write(output_path, result_audio.numpy(), 16000)
            temp_files.append(output_path)

            s3_url = save_audio_to_s3(job_id, output_path, target_language)
            response = {"job_id": job_id, "status": "success", "s3_url": s3_url,
                        "target_language": target_language, "gpu_memory": get_gpu_memory_usage()}

        return response
    except Exception as e:
        return {"error": str(e), "job_id": job_id, "status": "failed"}
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        clear_gpu_memory()
        gc.collect()

#initial
if __name__ == "__main__":
    print("🚀 Starting Dubbing API Endpoint...")
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        test_result = handler({"id": "test", "input": {"text": "Hello test", "target_language": "spa"}})
        print("Test Result:", json.dumps(test_result, indent=2))
