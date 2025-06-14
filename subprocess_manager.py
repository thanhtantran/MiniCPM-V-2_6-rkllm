import os
import time
import subprocess
import threading
import queue
import streamlit as st
import sys

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
            self.error_thread = threading.Thread(target=self._read_error, daemon=True)
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

    def _wait_for_ready(self):
        """Wait for the subprocess to be ready for inference"""
        print("Waiting for subprocess to be ready...")
        start_time = time.time()
        timeout = 180  # 180 seconds timeout
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"Process terminated during startup with return code: {self.process.returncode}")
                self._print_all_errors()
                return False
            
            # Check for ready signals in output
            try:
                output = self.output_queue.get(timeout=1)
                print(f"STARTUP OUTPUT: {output}")
                
                # Look for indicators that the process is ready
                if "Enter your input :" in output:
                    print("Subprocess is ready!")
                    self.is_ready = True
                    return True
                    
            except queue.Empty:
                # self._print_errors()
                continue
        
        print("Timeout waiting 3 minutes for subprocess to be ready")
        self._print_all_errors()
        return False
    
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
            
            # Collect ALL raw output until 'Enter your input :' marker
            all_raw_lines = []
            start_time = time.time()
            
            while time.time() - start_time < 180:  # 3 minute timeout
                # Check if process is still running
                if self.process.poll() is not None:
                    print(f"Process terminated during inference with return code: {self.process.returncode}")
                    self._print_all_errors()
                    break
                
                try:
                    output = self.output_queue.get(timeout=2)
                    print(f"RAW OUTPUT: {repr(output)}")
                    
                    # Check for end marker 'Enter your input :'
                    if "Enter your input :" in output:
                        print("Found 'Enter your input :' marker - response complete")
                        break
                    
                    # Add to raw collection (exclude the end marker)
                    all_raw_lines.append(output)
                        
                except queue.Empty:
                    self._print_errors()
                    continue
            
            if all_raw_lines:
                # Convert complete raw response to Markdown format
                raw_response = '\n'.join(all_raw_lines)
                markdown_response = self._convert_to_markdown(raw_response)
                return markdown_response
            else:
                return "**Error:** No response received"
                
        except Exception as e:
            print(f"Exception during inference: {e}")
            return f"**Error during inference:** {e}"
    
    def _convert_to_markdown(self, text):
        """Convert plain text response to Markdown format with proper formatting"""
        lines = text.split('\n')
        markdown_lines = []
        dash_line_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip the "Start vision inference..." line
            if "Start vision inference" in line:
                continue
                
            if line:
            # Check if this is a line of dashes (performance table border)
                if line.startswith('--------------------------------------------------------------------------------------'):
                    dash_line_count += 1
                    if dash_line_count == 1:
                        # First dash line - start code block
                        markdown_lines.append('```')
                        markdown_lines.append(line)
                    elif dash_line_count == 2:
                        # Second dash line - just add it
                        markdown_lines.append(line)
                    elif dash_line_count == 3:
                        # Third dash line - add it and end code block
                        markdown_lines.append(line)
                        markdown_lines.append('```')
                    else:
                        # Any additional dash lines
                        markdown_lines.append(line)
                # Handle timing information - separate lines
                elif "Vision encoder inference time:" in line: 
                    markdown_lines.append("**Vision Encoder Timing:**") 
                    markdown_lines.append(line) 
                    markdown_lines.append("") 
                elif "Time to first token:" in line: 
                    markdown_lines.append("**Generation Timing:**") 
                    markdown_lines.append(line) 
                    markdown_lines.append("") 
                # Handle the actual response content 
                elif line.startswith('"') and line.endswith('"'): 
                    markdown_lines.append("**Response:**") 
                    markdown_lines.append(line.strip('"')) 
                    markdown_lines.append("")
                else:
                    # Regular content
                    markdown_lines.append(line)
            else:
                # Add empty line for spacing (but not inside performance table)
                markdown_lines.append("")
        
        # Join lines and clean up formatting
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