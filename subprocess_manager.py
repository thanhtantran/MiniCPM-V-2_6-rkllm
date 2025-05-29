import os
import time
import subprocess
import threading
import queue
import streamlit as st

class StreamlitSubprocessManager:
    def __init__(self):
        self.process = None
        self.is_ready = False
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.input_queue = queue.Queue()
        
    def start_process(self):
        """Start the inference subprocess"""
        if self.process is not None:
            print("Process already running")
            return
        
        try:
            print("=== STARTING SUBPROCESS ===")
            print(f"Working directory: {os.getcwd()}")
            print(f"Python executable: {sys.executable}")
            print(f"Command: python multiprocess_inference.py")
            
            # Start subprocess with unbuffered output
            self.process = subprocess.Popen(
                [sys.executable, "-u", "multiprocess_inference.py"],  # -u for unbuffered
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True
            )
            
            print(f"Process started with PID: {self.process.pid}")
            
            # Start I/O threads
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.error_thread = threading.Thread(target=self._read_errors, daemon=True)
            self.input_thread = threading.Thread(target=self._write_input, daemon=True)
            
            self.output_thread.start()
            self.error_thread.start()
            self.input_thread.start()
            
            print("I/O threads started")
            
            # Wait for ready signals
            self._wait_for_ready()
            
        except Exception as e:
            print(f"Error starting process: {e}")
            self.process = None
    
    def _read_output(self):
        """Read output from the subprocess"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
        except Exception as e:
            print(f"Error reading output: {e}")
    
    def _read_error(self):
        """Read error output from the subprocess"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if line:
                    self.error_queue.put(line.strip())
        except Exception as e:
            print(f"Error reading stderr: {e}")
    
    def _print_errors(self):
        """Print any available error messages"""
        while not self.error_queue.empty():
            try:
                error = self.error_queue.get_nowait()
                print(f"STDERR: {error}")
            except queue.Empty:
                break
    
    def _print_all_errors(self):
        """Print all remaining error messages"""
        print("=== CHECKING FOR ERRORS ===")
        self._print_errors()
        
        # Also try to read any remaining stderr directly
        if self.process and self.process.stderr:
            try:
                remaining_stderr = self.process.stderr.read()
                if remaining_stderr:
                    print(f"REMAINING STDERR: {remaining_stderr}")
            except:
                pass
    
    def _write_input(self):
        """Write input to the subprocess"""
        try:
            while self.process and self.process.poll() is None:
                try:
                    input_text = self.input_queue.get(timeout=1)
                    if input_text is not None:  # Allow empty strings
                        print(f"SENDING TO SUBPROCESS: '{input_text}'")
                        self.process.stdin.write(input_text + '\n')
                        self.process.stdin.flush()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error writing input: {e}")
    
    def send_question(self, question, image_path):
        """Send a question to the inference process"""
        if not self.is_ready or not self.process:
            return "Error: Inference process not ready"
        
        try:
            print(f"=== SENDING QUESTION ===")
            print(f"Question: {question}")
            print(f"Image Path: {image_path}")
            
            # Send first line: Read the image in {{...}} carefully.
            image_check_line = f"Read the image in {{{{{image_path}}}}} carefully."
            print(f"First Line: {image_check_line}")
            self.input_queue.put(image_check_line)
            
            # Send second line: user question
            print(f"Second Line: {question}")
            self.input_queue.put(question)
            
            # Send three empty lines to signal end of input
            for _ in range(3):
                self.input_queue.put("")
            
            print("Image check line, question, and empty lines sent")
            
            # Collect ALL raw output with longer timeout and better detection
            all_raw_lines = []
            start_time = time.time()
            found_finished = False
            found_table_end = False
            
            while time.time() - start_time < 180:  # 3 minute timeout
                # Check if process is still running
                if self.process.poll() is not None:
                    print(f"Process terminated during inference with return code: {self.process.returncode}")
                    self._print_all_errors()
                    break
                
                try:
                    output = self.output_queue.get(timeout=2)  # Longer timeout
                    print(f"RAW OUTPUT: {repr(output)}")
                    
                    # Add to raw collection
                    all_raw_lines.append(output)
                    
                    # Look for (finished) marker
                    if "(finished)" in output:
                        found_finished = True
                        print("Found (finished) marker")
                    
                    # Look for end of performance table
                    if found_finished and "--------------------------------------------------------------------------------------" in output:
                        found_table_end = True
                        print("Found end of performance table")
                    
                    # Look for next input prompt after table
                    if found_table_end and "Enter your input" in output:
                        print("Found next input prompt - response complete")
                        break
                        
                except queue.Empty:
                    # If we found finished but no more output, wait a bit more
                    if found_finished:
                        print("Waiting for performance table...")
                        time.sleep(1)
                        continue
                    self._print_errors()
                    continue
            
            if all_raw_lines:
                # Remove the final "Enter your input" from display
                response_lines = []
                for line in all_raw_lines:
                    if "Enter your input" in line:
                        break
                    response_lines.append(line)
                
                if response_lines:
                    # Convert complete raw response to Markdown format
                    raw_response = '\n'.join(response_lines)
                    markdown_response = self._convert_to_markdown(raw_response)
                    return markdown_response
                else:
                    return "**Error:** No response content found"
            else:
                return "**Error:** No response received"
                
        except Exception as e:
            print(f"Exception during inference: {e}")
            return f"**Error during inference:** {e}"
    
    def _convert_to_markdown(self, text):
        """Convert plain text response to Markdown format"""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Keep the line as is for markdown
                markdown_lines.append(line)
            else:
                # Add empty line for spacing
                markdown_lines.append("")
        
        # Join lines and ensure proper markdown formatting
        result = '\n'.join(markdown_lines)
        
        # Clean up multiple consecutive empty lines
        import re
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result.strip()

    def stop_process(self):
        """Stop the inference process"""
        print("=== STOPPING SUBPROCESS ===")
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print(f"Process terminated with return code: {self.process.returncode}")
            except subprocess.TimeoutExpired:
                print("Process didn't terminate gracefully, killing...")
                self.process.kill()
            except Exception as e:
                print(f"Error stopping process: {e}")
            self.process = None
        self.is_ready = False