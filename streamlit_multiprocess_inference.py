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
    param.img_content = "<unk>".encode()
    
    extend_param = RKLLMExtendParam()
    extend_param.base_domain_id = 1
    param.extend_param = extend_param
    
    model_size = os.path.getsize(MODEL_PATH)
    print(f"Start loading language model (size: {model_size / 1024 / 1024:.2f} MB)")
    start_time = time.time()
    handle = init(param, result_callback)
    end_time = time.time()
    print(f"Language model loaded in {end_time - start_time:.2f} seconds")
    
    load_ready_queue.put("llm_ready")
    
    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value
    
    while True:
        prompt = prompt_queue.get()
        if prompt == "STOP":
            break
            
        image_embeddings = embedding_queue.get()
        if isinstance(image_embeddings, str) and image_embeddings == "ERROR":
            response_queue.put({"status": "error", "response": "Error processing image"})
            continue
            
        rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_MULTIMODAL,
                                        prompt=prompt,
                                        image_embed=image_embeddings)
        
        inference_start_time = time.time()
        run(handle, rkllm_input, infer_param, None)
    
    destroy(handle)

class StreamlitInferenceManager:
    def __init__(self):
        self.load_ready_queue = Queue()
        self.embedding_queue = Queue()
        self.img_path_queue = Queue()
        self.prompt_queue = Queue()
        self.response_queue = Queue()
        self.start_event = Event()
        
        self.vision_process = None
        self.lm_process = None
        self.is_ready = False
        
    def start_processes(self):
        """Start the vision and LLM processes"""
        self.vision_process = Process(target=vision_encoder_process,
                                    args=(self.load_ready_queue, self.embedding_queue, 
                                         self.img_path_queue, self.start_event))
        self.lm_process = Process(target=llm_process,
                                args=(self.load_ready_queue, self.embedding_queue, 
                                     self.prompt_queue, self.response_queue, self.start_event))
        
        self.vision_process.start()
        self.lm_process.start()
        
        # Wait for models to load
        ready_count = 0
        while ready_count < 2:
            status = self.load_ready_queue.get()
            print(f"Received ready signal: {status}")
            ready_count += 1
        
        print("All models loaded, ready for inference...")
        self.start_event.set()
        self.is_ready = True
        
    def send_question(self, question, image_path):
        """Send a question with image for inference"""
        if not self.is_ready:
            return "Error: Inference processes not ready"
            
        # Format the prompt
        image_placeholder = '<image_id>0</image_id><image>\n'
        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question.replace('{{' + str(image_path) + '}}', image_placeholder)}<|im_end|>
<|im_start|>assistant
"""
        
        # Send image and prompt
        self.img_path_queue.put(str(image_path))
        self.prompt_queue.put(prompt)
        
        # Wait for response
        try:
            response_data = self.response_queue.get(timeout=60)  # 60 second timeout
            if response_data["status"] == "success":
                return response_data["response"]
            else:
                return f"Error: {response_data['response']}"
        except:
            return "Error: Inference timeout"
            
    def stop_processes(self):
        """Stop the inference processes"""
        if self.is_ready:
            self.img_path_queue.put("STOP")
            self.prompt_queue.put("STOP")
            
        if self.vision_process:
            self.vision_process.join(timeout=5)
            if self.vision_process.is_alive():
                self.vision_process.terminate()
                
        if self.lm_process:
            self.lm_process.join(timeout=5)
            if self.lm_process.is_alive():
                self.lm_process.terminate()
                
        self.is_ready = False

if __name__ == "__main__":
    # For testing
    manager = StreamlitInferenceManager()
    manager.start_processes()
    
    try:
        while True:
            question = input("Enter question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            image_path = input("Enter image path: ")
            response = manager.send_question(question, image_path)
            print(f"Response: {response}")
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_processes()