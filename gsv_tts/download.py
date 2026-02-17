import os
import time
import logging
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path


def download_file(url, filename):
    logging.info(f"Downloading model from {url}")

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logging.error(f"ERROR: Download of {filename} incomplete or something went wrong. Expected {total_size_in_bytes} bytes, got {progress_bar.n} bytes.")
    else:
        logging.info(f"Download complete: {filename}")


def unzip_file(zip_filepath, extract_to):
    logging.info(f"Extracting {zip_filepath}...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Extraction complete, files located at: {extract_to}")


def check_latency(url, timeout=3):
    try:
        start_time = time.time()
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 405:
            response = requests.get(url, timeout=timeout, stream=True)
            response.close()
            
        end_time = time.time()
        
        if 200 <= response.status_code < 400:
            latency = (end_time - start_time) * 1000
            return True, latency
        else:
            return False, float('inf')
            
    except requests.RequestException:
        return False, float('inf')


def get_base_url():
    hf_url = "https://huggingface.co"
    ms_url = "https://www.modelscope.cn"
    
    hf_ok, hf_latency = check_latency(hf_url, timeout=5)
    ms_ok, ms_latency = check_latency(ms_url, timeout=5)
    
    if ms_ok and not hf_ok:
        logging.info("Selected ModelScope.")
        return "https://modelscope.cn/models/chinokiki"
        
    if hf_ok and not ms_ok:
        logging.info("Selected Hugging Face.")
        return "https://huggingface.co/cnmds"
    
    if not hf_ok and not ms_ok:
        logging.error("Both Hugging Face and ModelScope are unreachable. Defaulting to Hugging Face.")
        return "https://huggingface.co/cnmds"

    if ms_latency < hf_latency:
        logging.info("Selected ModelScope.")
        return "https://modelscope.cn/models/chinokiki"
    else:
        logging.info("Selected Hugging Face.")
        return "https://huggingface.co/cnmds"

base_url = None
def download_model(url, zip_filename):
    global base_url
    if base_url is None:
        base_url = get_base_url()
    if base_url == "https://modelscope.cn/models/chinokiki":
        url = base_url + url
    elif base_url == "https://huggingface.co/cnmds":
        url = base_url + url + "?download=true"

    download_file(url, zip_filename)
    unzip_file(zip_filename, os.path.dirname(zip_filename))
    os.remove(zip_filename)


def check_pretrained_models(models_dir):
    model_list = [
        Path(models_dir) / "chinese-hubert-base",
        Path(models_dir) / "fast_langdetect",
        Path(models_dir) / "g2p",
        Path(models_dir) / "sv",
    ]

    is_download = False
    for model_path in model_list:
        if not os.path.exists(model_path):
            is_download = True
            break
    
    if is_download:
        os.makedirs(models_dir, exist_ok=True)
        download_model(
            url="/GPTSoVITS-RT/resolve/master/pretrained_models2.zip",
            zip_filename=Path(models_dir) / "pretrained_models2.zip"
        )