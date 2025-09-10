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
            "max_cached": torch.cuda.max_memory_reserved() / 1024**3,
        }
    return {}

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_audio_to_s3(job_id, file_path, target_language):
    if not S3_BUCKET:
        logger.warning("S3_BUCKET_NAME environment variable not set. Skipping S3 upload.")
        return None

    try:
        s3_key = f"{S3_OUTPUT_DIR}/{job_id}_{target_language}_{os.path.basename(file_path)}"
        s3.upload_file(file_path, S3_BUCKET, s3_key)
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        logger.info(f"Successfully uploaded {file_path} to S3 at {s3_url}")
        return s3_url
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to S3: {e}")
        return None

# ------------------- MAIN MODEL FUNCTIONS ------------------- #
def load_model():
    """Lazy loads the model and processor into global variables."""
    global model, processor, device
    if model is None:
        logger.info("⏳ Loading model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if COMPUTE_TYPE == "float16" and torch.cuda.is_available() else torch.float32

        model_name = "facebook/seamless-m4t-v2-large"
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR
        )

        model = SeamlessM4Tv2Model.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR
        )
        logger.info("✅ Model loaded successfully!")

def speech_to_text_translation(audio_path, source_language, target_language):
    """Transcribes and translates speech from an audio file."""
    load_model()
    try:
        audio, _ = sf.read(audio_path, dtype='float32')
        # Here we add the padding=True argument to prevent the error
        inputs = processor(audios=audio, return_tensors="pt", padding=True).to(device)
        
        output_tokens = model.generate(
            **inputs,
            tgt_lang=target_language,
            generate_speech=False,
            sp_model_kwargs={"tgt_lang": target_language}
        )

        translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        logger.error(f"Speech-to-text translation failed: {e}")
        raise

def text_to_speech_dubbing(text, target_language):
    """Generates speech from a given text."""
    load_model()
    try:
        # Here we add the padding=True argument to prevent the error
        text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        audio_output = model.generate(**text_inputs, tgt_lang=target_language, generate_speech=True, )
        result_audio = audio_output[0].cpu().numpy().squeeze()
        return result_audio
    except Exception as e:
        logger.error(f"Text-to-speech dubbing failed: {e}")
        raise

def dubbing_pipeline(job_input):
    """Main pipeline function to handle both translation and dubbing."""
    # job_id = runpod.get_job_id()

    if not job.get("id"):
        return {"error": "job id not found"}
        
    job_id = job["id"]
    
    temp_files = []
    
    try:
        input_data = job_input['input']
        audio_url = input_data.get("audio_url")
        text = input_data.get("text")
        source_language = input_data.get("source_language")
        target_language = input_data.get("target_language")

        if not target_language:
            raise ValueError("target_language must be provided.")
        
        if not (audio_url or text):
            raise ValueError("Either 'audio_url' or 'text' must be provided.")

        if audio_url:
            local_audio_path = f"/tmp/{uuid.uuid4()}_input.mp3"
            subprocess.run(["wget", "-O", local_audio_path, audio_url], check=True)
            temp_files.append(local_audio_path)
            
            translated_text = speech_to_text_translation(local_audio_path, source_language, target_language)
            result_audio = text_to_speech_dubbing(translated_text, target_language)

            output_path = f"/tmp/{uuid.uuid4()}_dubbed.wav"
            sf.write(output_path, result_audio.numpy(), 16000)
            temp_files.append(output_path)

            s3_url = save_audio_to_s3(job_id, output_path, target_language)
            response = {"job_id": job_id, "status": "success", "s3_url": s3_url,
                        "translated_text": translated_text, "target_language": target_language,
                        "gpu_memory": get_gpu_memory_usage()}
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
    load_model()
    runpod.serverless.start({"handler": dubbing_pipeline})