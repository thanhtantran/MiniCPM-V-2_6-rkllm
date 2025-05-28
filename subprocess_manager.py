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
        
    def start_inference_process(self):
        """Start the multiprocess_inference.py as a subprocess"""
        try:
            print("=== STARTING SUBPROCESS ===")
            print(f"Working directory: {os.getcwd()}")
            print(f"Python executable: {os.sys.executable}")
            print("Command: python multiprocess_inference.py")
            
            # Start the subprocess
            self.process = subprocess.Popen(
                ['python', 'multiprocess_inference.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd()  # Explicitly set working directory
            )
            
            print(f"Process started with PID: {self.process.pid}")
            
            # Start threads to handle I/O
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.error_thread = threading.Thread(target=self._read_error, daemon=True)
            self.input_thread = threading.Thread(target=self._write_input, daemon=True)
            
            self.output_thread.start()
            self.error_thread.start()
            self.input_thread.start()
            
            print("I/O threads started")
            
            # Wait for the process to be ready
            start_time = time.time()
            ready_signals = 0
            input_prompt_seen = False
            
            while time.time() - start_time < 120:  # 2 minute timeout
                # Check if process is still running
                if self.process.poll() is not None:
                    print(f"Process terminated early with return code: {self.process.returncode}")
                    # Print any error output
                    self._print_all_errors()
                    return False
                
                try:
                    output = self.output_queue.get(timeout=1)
                    print(f"STDOUT: {output}")
                    
                    # Check for ready signals
                    if "vision_ready" in output or "llm_ready" in output:
                        ready_signals += 1
                        
                    # Check for the interactive mode message
                    if "All models loaded" in output:
                        print("Models loaded message detected")
                        
                    # Check for the input prompt
                    if "Enter your input" in output:
                        input_prompt_seen = True
                        print("Input prompt detected")
                        
                    # If we've seen both ready signals and the input prompt, we're ready
                    if ready_signals >= 2 and input_prompt_seen:
                        self.is_ready = True
                        print("=== SUBPROCESS READY ===")
                        return True
                        
                except queue.Empty:
                    # Check for errors
                    self._print_errors()
                    continue
            
            print("Timeout waiting for subprocess to be ready")
            self._print_all_errors()
            return False
            
        except Exception as e:
            print(f"Exception starting subprocess: {e}")
            st.error(f"Failed to start inference process: {e}")
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
            
            # Send three empty lines to signal end of input (as expected by original script)
            for _ in range(3):
                self.input_queue.put("")
            
            print("Image check line, question, and empty lines sent")
            
            # Collect the response
            all_lines = []
            start_time = time.time()
            finished = False
            
            while time.time() - start_time < 120:  # 120 second timeout
                # Check if process is still running
                if self.process.poll() is not None:
                    print(f"Process terminated during inference with return code: {self.process.returncode}")
                    self._print_all_errors()
                    break
                
                try:
                    output = self.output_queue.get(timeout=1)
                    print(f"RAW OUTPUT: {repr(output)}")
                    
                    # Collect all lines until we find the performance table end
                    all_lines.append(output)
                    
                    # Check if we've reached the end of the performance table
                    if "--------------------------------------------------------------------------------------" in output and len(all_lines) > 5:
                        # We've collected the complete response including the table
                        finished = True
                        print("Found end of performance table - response complete")
                        break
                        
                except queue.Empty:
                    self._print_errors()
                    continue
            
            if not finished:
                print("Warning: Response collection ended without finding complete response")
            
            if all_lines:
                # Parse the response to extract only the answer
                answer_lines = self._extract_answer_from_response(all_lines)
                if answer_lines:
                    # Convert response to Markdown format
                    markdown_response = self._convert_to_markdown('\n'.join(answer_lines))
                    return markdown_response
                else:
                    return "**Error:** Could not extract answer from response"
            else:
                return "**Error:** No response received"
                
        except Exception as e:
            print(f"Exception during inference: {e}")
            return f"**Error during inference:** {e}"
    
    def _extract_answer_from_response(self, all_lines):
        """Extract only the answer portion from the complete subprocess response"""
        answer_lines = []
        skip_initial_status = True
        found_answer_start = False
        
        for line in all_lines:
            line = line.strip()
            
            # Skip empty lines at the beginning
            if not line and not found_answer_start:
                continue
            
            # Skip the initial status lines (for first response)
            if skip_initial_status:
                if any(status in line for status in [
                    "Start vision inference",
                    "Vision encoder inference time",
                    "Time to first token"
                ]):
                    continue
                else:
                    skip_initial_status = False
            
            # Stop when we hit the (finished) marker
            if "(finished)" in line:
                print("Found (finished) marker - stopping answer collection")
                break
            
            # Stop when we hit the performance table separator
            if "--------------------------------------------------------------------------------------" in line:
                print("Found performance table - stopping answer collection")
                break
            
            # Skip debug/status messages
            if any(skip in line.lower() for skip in [
                'loading', 'input:', 'formatted prompt', 'enter your', 'enter another'
            ]):
                continue
            
            # This is part of the answer
            if line:
                found_answer_start = True
                answer_lines.append(line)
        
        print(f"Extracted answer lines: {answer_lines}")
        return answer_lines
    
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