import faulthandler
faulthandler.enable()
import os
import time
import signal
from multiprocessing import Process, Queue, Event
import cv2
import numpy as np
from rkllm_binding import *
from rknnlite.api.rknn_lite import RKNNLite
import sys
import json
import re

# Vision encoder process (same as original)
def vision_encoder_process(load_ready_queue, embedding_queue, img_path_queue, start_event):
    VISION_ENCODER_PATH = "model/vision_transformer.rknn"
    img_size = 448
    
    vision_encoder = RKNNLite(verbose=False)
    model_size = os.path.getsize(VISION_ENCODER_PATH)
    print(f"Start loading vision encoder model (size: {model_size / 1024 / 1024:.2f} MB)")
    start_time = time.time()
    vision_encoder.load_rknn(VISION_ENCODER_PATH)
    end_time = time.time()
    print(f"Vision encoder loaded in {end_time - start_time:.2f} seconds")
    vision_encoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    
    load_ready_queue.put("vision_ready")
    start_event.wait()
    
    def process_image(img_path, vision_encoder):
        img = cv2.imread(img_path)
        if img is None:
            return None
        print("Start vision inference...")
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]
        
        start_time = time.time()
        image_embeddings = vision_encoder.inference(inputs=[img], data_format="nhwc")[0].astype(np.float32)
        end_time = time.time()
        print(f"Vision encoder inference time: {end_time - start_time:.2f} seconds")
        return image_embeddings

    while True:
        img_path = img_path_queue.get()
        if img_path == "STOP":
            break
        embeddings = process_image(img_path, vision_encoder)
        if embeddings is not None:
            embedding_queue.put(embeddings)
        else:
            embedding_queue.put("ERROR")

# LLM process with response collection
def llm_process(load_ready_queue, embedding_queue, prompt_queue, response_queue, start_event):
    MODEL_PATH = "model/qwen.rkllm"
    handle = None
    
    def signal_handler(signal, frame):
        print("Ctrl-C pressed, exiting...")
        global handle
        if handle:
            abort(handle)
            destroy(handle)
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    os.environ["RKLLM_LOG_LEVEL"] = "1"
    
    # Collect response text
    current_response = []
    inference_count = 0
    inference_start_time = 0
    
    def result_callback(result, userdata, state):
        nonlocal inference_start_time, inference_count, current_response
        if state == LLMCallState.RKLLM_RUN_NORMAL:
            if inference_count == 0:
                first_token_time = time.time()
                print(f"Time to first token: {first_token_time - inference_start_time:.2f} seconds")
            inference_count += 1
            token = result.contents.text.decode()
            current_response.append(token)
            print(token, end="", flush=True)
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            print("\n\n(finished)")
            full_response = "".join(current_response)
            response_queue.put({"status": "success", "response": full_response})
            current_response = []
            inference_count = 0
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            print("\nError occurred during LLM call")
            response_queue.put({"status": "error", "response": "Error occurred during inference"})
            current_response = []
            inference_count = 0
    
    # Initialize LLM
    param = create_default_param()
    param.model_path = MODEL_PATH.encode()
    param.img_start = "<image>".encode()
    param.img_end = "</image>".encode()
    param.img_content = "