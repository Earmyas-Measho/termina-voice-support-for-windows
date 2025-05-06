import patch_ctypes
import sys
import os
import subprocess
import json
import tempfile
import time
import threading
import ctypes
import ctypes.util
import whisper
import numpy as np
import pyaudio
import wave
import requests
import re
import importlib.util
import platform
import pickle
from collections import deque
import configparser
import csv
from datetime import datetime
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DANGEROUS_PATTERNS = [
    r'\brm\b',          # Unix/Linux remove
    r'\bdel\b',         # Windows delete
    r'\berase\b',       # Windows delete alias
    r'Remove-Item',     # PowerShell remove
    r'\brd\b',          # Windows remove directory
    r'\brmdir\b',       # Unix/Windows remove directory
    r'\bformat\b',      # Formatting drives
    r'\bmkfs\b',        # Linux make filesystem
    r'fdisk',           # Disk partitioning
    r'diskpart',        # Windows disk partitioning
    r'sudo\b',          # Running as root/admin (use with caution)
    r'runas\b',         # Windows run as different user
    r'Invoke-Expression',# PowerShell code execution
    r'\biex\b',         # PowerShell Invoke-Expression alias
    r'curl\s+.*\|.*sh', # Downloading and executing scripts (common pattern)
    r'wget\s+.*\|.*sh', # Downloading and executing scripts (common pattern)
    r'Invoke-WebRequest\s+.*\|.*iex', # PowerShell download and execute
    r'iwr\s+.*\|.*iex', # PowerShell iwr alias
    r':\(\)\s*\{.*\}\s*;\s*:', # Fork bomb pattern
    r'\bshutdown\b',    # System shutdown/reboot
    r'\breboot\b',      # System reboot
    r'Stop-Computer',   # PowerShell shutdown/reboot
    r'Restart-Computer',# PowerShell reboot
    r'>\s*/dev/null',   # Redirecting to null device (can hide errors) - adjust if needed
    # Add more patterns as needed
]

# Compile regex patterns for efficiency
DANGEROUS_REGEX = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]

def is_command_dangerous(command):
    """Checks if a command matches any pattern in the blacklist."""
    for pattern in DANGEROUS_REGEX:
        if pattern.search(command):
            return True
    return False

# List and select microphone
def select_microphone():
    mic = pyaudio.PyAudio()
    print("\n Available Microphones:")
    
    # Get device count
    info = mic.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount', 0)
    
    # Fix: Ensure numdevices is an integer
    if isinstance(numdevices, str):
        try:
            numdevices = int(numdevices)
        except ValueError:
            numdevices = 0
    
    for i in range(0, numdevices):
        max_input_channels = mic.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels', 0)
        # Ensure max_input_channels is an integer and not None
        if max_input_channels is None:
            max_input_channels = 0
        elif isinstance(max_input_channels, str):
            try:
                max_input_channels = int(max_input_channels)
            except ValueError:
                max_input_channels = 0
                
        if max_input_channels > 0:
            info = mic.get_device_info_by_index(i)
            print(f"{i}: {info['name']}")
            
    # Add a prompt to skip
    print("\n(Press Enter to skip voice input and use Text mode only)")
    choice = input("🔧 Enter the device index for your microphone: ").strip()

    if not choice: # User pressed Enter
        print("Skipping microphone selection. Voice input will be disabled.")
        return None

    try:
        index = int(choice)
        if 0 <= index < numdevices:
             # Further check if it's an input device (optional but good)
             device_info = mic.get_device_info_by_index(index)
             if device_info.get('maxInputChannels', 0) > 0:
                 print(f"Selected microphone: {device_info['name']}")
                 return index
             else:
                 print(f"Device {index} is not an input device.")
                 return None
        else:
            print(f"Invalid index. Please choose from the listed devices.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None
    finally:
        # Terminate PyAudio instance if created
        if mic:
            mic.terminate()

whisper_model = None

# Store command history
HISTORY_FILE = os.path.join(SCRIPT_DIR, "command_history.pkl")
MAX_HISTORY = 20
command_history = deque(maxlen=MAX_HISTORY)

def load_command_history():
    """Load command history from file."""
    global command_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'rb') as f:
                command_history = pickle.load(f)
            print(f"Loaded {len(command_history)} commands from history.")
    except Exception as e:
        print(f"Could not load command history: {e}")
        command_history = deque(maxlen=MAX_HISTORY)

def save_command_history():
    """Save command history to file."""
    try:
        with open(HISTORY_FILE, 'wb') as f:
            pickle.dump(command_history, f)
    except Exception as e:
        print(f"Could not save command history: {e}")

# Add a global variable
whisper_model = None

# Replace the model loading in main() with this pattern
def load_whisper_model(config):
    """Lazy load the Whisper model only when needed."""
    global whisper_model
    
    if whisper_model is None:
        model_size = config.get('Whisper', 'model_size', fallback='small')
        print(f"Loading Whisper '{model_size}' model (more accurate, may take longer)...")
        try:
            whisper_model = whisper.load_model(model_size)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
    
    return whisper_model

# Modify get_voice_command() to use lazy loading
def get_voice_command(input_device_index=None, config=None):
    """Records audio and transcribes it using Whisper."""
    if config is None:
        config = load_config()
    
    # Use config values instead of hardcoded constants
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = config.getint('Recording', 'rate', fallback=16000)
    CHUNK = config.getint('Recording', 'chunk_size', fallback=4096)
    RECORD_SECONDS = config.getint('Recording', 'max_record_seconds', fallback=12)
    SILENCE_THRESHOLD = config.getint('Recording', 'silence_threshold', fallback=100)
    SILENCE_DURATION = config.getfloat('Recording', 'silence_duration', fallback=3.0)
    
    # Verify microphone before proceeding - IMPROVED VERSION
    audio = pyaudio.PyAudio()
    try:
        # Better microphone test with user prompt
        print("\n📢 Quick mic check... please say something now")
        
        # Create test stream
        test_stream = audio.open(format=FORMAT, channels=CHANNELS,
                          rate=RATE, input=True,
                          input_device_index=input_device_index,
                          frames_per_buffer=CHUNK)
        
        # Record for a short period to get a proper sample
        test_frames = []
        test_max_volume = 0
        
        # Collect 1 second of audio for testing
        for _ in range(int(RATE / CHUNK)):
            test_data = test_stream.read(CHUNK, exception_on_overflow=False)
            test_frames.append(test_data)
            
            # Check volume level
            test_audio = np.frombuffer(test_data, dtype=np.int16)
            test_peak = np.abs(test_audio).max() / 32767.0
            test_max_volume = max(test_max_volume, test_peak)
            
            # Show simple volume meter
            bar = "█" * int(test_peak * 30)
            print(f"\rMic level: {bar}", end="")
        
        test_stream.stop_stream()
        test_stream.close()
        
        print(f"\nMicrophone check complete. Peak level: {test_max_volume:.6f}")
        
        # Only warn if level is extremely low
        if test_max_volume < 0.005:  # Was 0.001, now less sensitive
            print("⚠️ Warning: Microphone appears to be very quiet or muted.")
            print("Check your Windows sound settings and permissions.")
        else:
            print("✅ Microphone is working.")
    except Exception as e:
        print(f"⚠️ Warning: Microphone test error: {e}")

    # Clearer countdown instructions with more time
    print("\n🎙️ VOICE COMMAND INSTRUCTIONS:")
    print("1. Prepare to speak after the countdown")
    print("2. Speak clearly after 'GO!'")
    print("3. You'll have 12 seconds, but recording will stop after 3s of silence")
    print("\n🎤 Get ready to speak in:")
    
    # Slower countdown for better preparation
    for i in range(3, 0, -1):
        print(f"\r{i}...", end="", flush=True)
        time.sleep(1.0)  # Full 1-second pauses
    
    print("\r🗣️ GO! SPEAK NOW...                           ")

    # Start the actual recording stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    max_volume_this_recording = 0
    recording_started = False
    
    # Define colors for a prettier bar (ANSI color codes)
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    # Define gradient block characters for smoother visualization
    blocks = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']

    start_time = time.time()
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Volume calculation
            audio_data = np.frombuffer(data, dtype=np.int16)
            peak = np.abs(audio_data).max() / 32767.0
            norm = np.linalg.norm(audio_data) / (32767.0 * np.sqrt(len(audio_data)))
            volume_norm = max(peak, norm * 15)  # Increased amplification for visibility
            max_volume_this_recording = max(max_volume_this_recording, volume_norm)

            # Mark when recording actually starts detecting sound
            if not recording_started and volume_norm > 0.01:
                recording_started = True
                print("\r💬 Sound detected! Recording...                    ")

            # Bar configuration - same fancy bar
            bar_width = 30
            full_blocks = int(volume_norm * bar_width)
            remainder = (volume_norm * bar_width) - full_blocks
            partial_block_idx = min(int(remainder * 8), 7)
            
            # Create the colorful bar
            bar_str = ""
            for j in range(bar_width):
                if j < bar_width * 0.25:
                    color = GREEN
                elif j < bar_width * 0.5:
                    color = BLUE
                elif j < bar_width * 0.75:
                    color = YELLOW
                else:
                    color = RED
                
                if j < full_blocks:
                    bar_str += f"{color}█{RESET}"
                elif j == full_blocks and partial_block_idx > 0:
                    bar_str += f"{color}{blocks[partial_block_idx]}{RESET}"
                else:
                    bar_str += "░"
            
            elapsed_time = time.time() - start_time
            remaining = RECORD_SECONDS - elapsed_time
            timer_str = f"{elapsed_time:.1f}s/{remaining:.1f}s left"
            
            # Volume display with threshold indicator
            threshold_marker = f"[{'-'*10}|{'-'*10}]"
            
            vol_level = "LOUD! 🔊" if volume_norm > 0.5 else "Good 🎤" if volume_norm > 0.1 else "Quiet 🔈" if volume_norm > 0.01 else "Silent 🔇"
            
            output_line = f"Recording: [{bar_str}] {timer_str} ({vol_level})"
            padding = " " * 5
            print(f"\r{output_line}{padding}", end="", flush=True)

            # Don't start silence detection until 1 second has passed
            if elapsed_time < 1.0:
                continue
                
            # Less aggressive silence detection
            if norm < (SILENCE_THRESHOLD / 32767.0):
                silent_chunks += 1
                # Visual indicator scaled to silence duration
                if silent_chunks % 3 == 0:
                    silence_sec = (silent_chunks * CHUNK / RATE)
                    silence_percent = int(min(100, (silence_sec / SILENCE_DURATION) * 100))
                    print(f"\r{output_line} 🔇 {silence_percent}%{padding}", end="", flush=True)
            else:
                silent_chunks = 0  # Reset if sound detected

            # Only stop after at least 3 seconds AND if we've detected any audio
            if elapsed_time > 3.0 and recording_started and (silent_chunks * CHUNK / RATE) >= SILENCE_DURATION:
                print("\n✅ Detected extended silence after speech, stopping recording.")
                break

        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("\nWarning: Input overflowed. Skipping frame.", end="")
            else:
                raise

    print("\nProcessing audio...")
    print(f"Max volume during recording: {max_volume_this_recording:.4f}")
    
    # More helpful feedback when volume is too low
    if max_volume_this_recording < 0.01:
        print("\n❌ ERROR: Extremely low volume detected.")
        print("Your microphone appears to be muted or not working properly.")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Check if your microphone is physically muted (switch or button)")
        print("2. Open Windows Sound Settings and ensure the correct device is enabled")
        print("3. Try option (D) for microphone diagnostics")
        print("4. Try option (M) to test this specific microphone")
        return None
    elif max_volume_this_recording < 0.05:
        print("\n⚠️ WARNING: Very low volume detected.")
        print("Your voice may not have been captured clearly.")
        
    # Rest of the function remains the same...
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_audio_path = tmp_file.name

    # Write to wave file
    wf = wave.open(tmp_audio_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Transcribe using Whisper
    start_transcribe = time.time()
    try:
        # Use the loaded model, specify language as English
        result = whisper_model.transcribe(tmp_audio_path, language="en", fp16=False) # fp16=False for CPU
        transcribed_text = result["text"].strip()
        print(f"You said: {transcribed_text}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        transcribed_text = "" # Return empty string on error
    finally:
        # Clean up the temporary file
        try:
            os.remove(tmp_audio_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")

    end_transcribe = time.time()
    print(f"Processing time: {end_transcribe - start_transcribe:.2f} seconds")

    # Check if transcription is empty or just punctuation/whitespace
    if not transcribed_text or transcribed_text.translate(str.maketrans('', '', '.,!?;: ')).isalnum() == False:
         print("Could not understand audio or transcription was empty.")
         return None # Return None if transcription failed or is empty

    # Initialize metrics for voice input
    metrics = {
        'input_mode': 'voice',
        'command_type': 'single',
        'transcription_accuracy': 0,  # We don't know actual accuracy
        'command_length': len(transcribed_text)
    }

    return transcribed_text, metrics  # Return metrics with the command

# Update detect_shell function
def detect_shell():
    """Detect the Windows shell (PowerShell or CMD). Returns 'unsupported' for other OS."""
    system = platform.system().lower()

    if system == "windows":
        # Check if PowerShell is available, otherwise use CMD
        try:
            # Try to run a simple PowerShell command
            result = subprocess.run(["powershell", "-Command", "echo test"],
                                  capture_output=True, text=True, check=False, shell=True) # Added shell=True for safety
            if result.returncode == 0:
                return "powershell"
        except Exception:
            # If PowerShell check fails, assume CMD
            pass
        return "cmd"
    else:
        # Operating system is not Windows
        return "unsupported"

# Update get_system_prompt function to include guidance on paths
def get_system_prompt():
    shell = detect_shell()

    if shell == "cmd":
        return (
            "You are a command-line assistant that converts natural language instructions into "
            "Windows CMD commands. Respond with a numbered list of 3 different commands that accomplish "
            "the user's request, from simplest to most advanced. Format each command with an emoji and explanation:\n\n"
            "1. Basic Command\n"
            "    📝 `dir` - Lists files in the current directory\n\n"
            "2. Standard Command\n"
            "    📝 `dir /b` - Lists files in brief format\n\n"
            "3. Advanced Command\n"
            "    📝 `dir /a /s` - List all files recursively including hidden ones\n\n"
            "Always include the actual command in backticks (`) within your explanation. "
            "Ensure all commands are valid CMD syntax. Do not use redirection symbols (like > or |) unless specifically requested. "
            "Keep the commands focused on displaying output in the terminal by default.\n\n"
            "IMPORTANT: Do not use placeholder paths like 'C:\\path\\to\\directory'. "
            "Instead, use the current directory (.), user profile (%USERPROFILE%), "
            "or system-defined paths (%WINDIR%, %PROGRAMFILES%)."
        )
    elif shell == "powershell":
        return (
            "You are a command-line assistant that converts natural language instructions into "
            "Windows PowerShell commands. Respond with a numbered list of 3 different commands that accomplish "
            "the user's request, from simplest to most advanced. Format each command with an emoji and explanation:\n\n"
            "1. Basic Command\n"
            "   `Get-ChildItem` - Lists files and folders in the current directory\n\n"
            "2. Standard Command\n"
            "   `Get-ChildItem -Name` - Lists only the names of files and folders\n\n"
            "3. Advanced Command\n"
            "   `Get-ChildItem -Recurse -Force -File | Select-Object FullName, Length, LastWriteTime` - Lists all files recursively (including hidden) with details\n\n"
            "Always include the actual command in backticks (`) within your explanation. "
            "Ensure all commands are valid PowerShell syntax. Use PowerShell cmdlets (e.g., Get-ChildItem, Select-Object) where appropriate. "
            "Keep the commands focused on displaying output in the terminal by default.\n\n"
            "IMPORTANT: Do not use placeholder paths like 'C:\\path\\to\\search'. "
            "Instead, use the current directory (.), user's home directory ($HOME), "
            "or the current working directory parameter: $(Get-Location)."
        )
    else: # Handle unsupported case
        return "This operating system is not supported. This tool only works on Windows."

# Modify process_with_mistral to use get_system_prompt instead of hardcoded prompt
def process_with_mistral(command, shell, force_chain=False, debug=False):
    """Process command with Mistral AI, supporting command chaining."""
    # Always load fresh config
    config = load_config()
    
    # Get API key from config (try both sections for compatibility)
    api_key = config.get('API', 'mistral_api_key', fallback='')
    endpoint = config.get('API', 'mistral_endpoint', fallback='https://api.mistral.ai/v1/chat/completions')
    model = config.get('API', 'mistral_model', fallback='mistral-tiny')
    
    # If key not found, check environment variable
    if not api_key:
        api_key = os.environ.get('MISTRAL_API_KEY', '')
    
    # If still empty, prompt user
    if not api_key:
        print("\n⚠️ Mistral API key not found in config or environment variables.")
        api_key = input(" Please enter your Mistral API key (or press Enter to cancel): ").strip()
        if not api_key:
            print(" Operation cancelled.")
            return None
        
        # Save to config for future use
        config['API']['mistral_api_key'] = api_key
        with open(os.path.join(SCRIPT_DIR, "config.ini"), 'w') as f:
            config.write(f)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use other config values
    model_name = config.get('Mistral', 'model', fallback='mistral-large-latest')
    temperature = config.getfloat('Mistral', 'temperature', fallback=0.3)
    max_tokens = config.getint('Mistral', 'max_tokens', fallback=800)
    
    # Check if the command might need a chain of commands or if force_chain is True
    chain_keywords = ['and then', 'after that', 'followed by', 'next', 'create and move', 
                     'create and then', 'first', 'download and extract', 'combine', 'sequence']
    needs_chain = force_chain or any(keyword in command.lower() for keyword in chain_keywords)
    
    # Create a specialized prompt for command chains
    if needs_chain:
        system_prompt = (
            "You are a command-line assistant that converts complex tasks into a sequence of "
            f"individual {shell.upper()} commands. For multi-step operations, provide each command "
            "with a clear COMMAND: prefix.\n\n"
            "Format your response like this:\n"
            "Step 1:\nCOMMAND: [first command]\n[explanation of what this step does]\n\n"
            "Step 2:\nCOMMAND: [second command]\n[explanation of what this step does]\n\n"
            f"Use only valid {shell.upper()} syntax. Do not use placeholder paths - use the current "
            "directory (.) or absolute paths when necessary."
        )
        
        user_content = f"Convert this complex task into a sequence of {shell.upper()} commands: {command}"
    else:
        # Use the standard system prompt for single commands
        system_prompt = get_system_prompt()
        user_content = f"Generate {shell.upper()} commands for: '{command}'"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    print(f"\n Consulting Mistral AI for command suggestions...")
    
    start_time = time.time()
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f" Received suggestions in {processing_time:.2f} seconds")
        
        # Debug mode - show the raw response
        if debug:
            print("\n DEBUG - Raw Mistral Response:")
            content = response_json["choices"][0]["message"]["content"]
            print(content)
            print("\n --- End of raw response ---\n")
        
        # Pass whether this is a chain request to the parser
        return parse_mistral_response(response_json, command, needs_chain)
    except Exception as e:
        print(f" Error during Mistral API request: {e}")
        return None

# Modify execute_command confirmation for clarity
def execute_command(command, input_mode="text"):
    """Execute a command in the appropriate shell."""
    shell = detect_shell()
    if shell == "unsupported":
        print("\nError: Cannot execute command. This tool only supports Windows (CMD/PowerShell).")
        return

    # Initialize metrics dictionary
    metrics = {
        'input_mode': input_mode,
        'command_type': 'single',  
        'command_length': len(command),
        'required_correction': False
    }

    print(f"\n Command Ready to Execute ({shell.upper()}):")
    # Display command clearly, maybe wrap long ones
    if len(command) > 80:
         print(f"  {command[:77]}...")
         print(f"  Full command: {command}")  # Always show full command
    else:
         print(f"  {command}")

    # Check for potentially slow commands and warn user
    slow_patterns = ['-Recurse', '-Force', '-Depth', '/s']
    is_potentially_slow = any(pattern in command for pattern in slow_patterns)
    
    if is_potentially_slow:
        print("\n⚠️ NOTE: This command might take a long time to run if there are many files.")
        print("      Press Ctrl+C at any time to cancel execution.")

    # Emphasize checking the command
    print("\n⚠️ IMPORTANT: Review the command carefully before executing! ⚠️")
    confirmation = input("Execute this command? (Y)es / (N)o / (E)dit / (L)imit results / e(X)plain: ").strip().lower()

    if confirmation == 'l':
        if shell == "powershell":
            # Add a limit for PowerShell commands
            if 'Get-ChildItem' in command and '-Recurse' in command:
                modified_command = command + " | Select-Object -First 50"
                print(f"\n Modified command with limit: {modified_command}")
                execute_modified = input("Execute this modified command? (Y/N): ").strip().lower()
                if execute_modified != 'y':
                    print("\n Command cancelled")
                    return
                command = modified_command
            else:
                print("Limit option is currently only available for recursive Get-ChildItem commands.")
                return
        else:
            print("Limit option is currently only available for PowerShell commands.")
            return
        
    if confirmation == 'y' or confirmation == 'l':
        # Command is executed as is
        metrics['required_correction'] = False
        
        # Add to history if it's actually executed
        command_history.appendleft(command)
        save_command_history()
        
        print("\n Executing command...")
        
        start_time = time.time()
        success = False
        
        try:
            # Use the appropriate execution method based on shell
            if shell == "powershell":
                process = subprocess.Popen(["powershell", "-Command", command], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                          text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # cmd
                process = subprocess.Popen(["cmd", "/c", command], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            # Add this section for handling Ctrl+C
            try:
                stdout, stderr = process.communicate(timeout=600)  # 10-minute timeout
            except KeyboardInterrupt:
                # Handle Ctrl+C properly
                print("\n\n⚠️ Command execution interrupted by user")
                process.terminate()
                
                # Wait a bit for process to terminate
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate
                
                # Update metrics to show cancellation
                metrics['was_successful'] = False
                metrics['execution_time'] = time.time() - start_time
                log_metrics(metrics)
                return
            
            end_time = time.time()
            exec_time = end_time - start_time
            success = process.returncode == 0
            
            # Save metrics
            metrics['execution_time'] = exec_time
            metrics['was_successful'] = success
            log_metrics(metrics)
            
            # Rest of your existing code...
            if stdout:
                print(stdout)
                # For large outputs, show how many lines were returned
                line_count = stdout.count('\n')
                if line_count > 10:
                    print(f"\n Total: {line_count} lines of output")
            else:
                print("\n ℹ️ The command executed successfully but returned no results.")
                
            print(f"\n Command execution complete (took {exec_time:.2f} seconds)")
        except subprocess.TimeoutExpired:
            print("\n⚠️ Command timed out after 60 seconds. Consider refining your command or using option (L) to limit results.")
        except subprocess.CalledProcessError as e:
            print(f"\nError during command execution: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
        except KeyboardInterrupt:
            print("\n\n⚠️ Command execution cancelled by user.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
    elif confirmation == 'e':
        edited = input("\n Enter modified command: ").strip()
        if edited:
            print("\n Executing modified command...\n")
            try:
                # Reuse the same logic but with the edited command
                start_time = time.time()
                if shell == "powershell":
                    result = subprocess.run(["powershell", "-Command", edited], 
                                          check=True, shell=True,
                                          capture_output=True, text=True)
                    output = result.stdout
                elif shell == "cmd":
                    result = subprocess.run(edited, check=True, shell=True,
                                          capture_output=True, text=True)
                    output = result.stdout
                
                elapsed_time = time.time() - start_time
                
                # Display the output or a message if empty
                if output.strip():
                    print(output)
                else:
                    print("\n ℹ️ The command executed successfully but returned no results.")
                    
                print(f"\n Command execution complete (took {elapsed_time:.2f} seconds)")
            except subprocess.CalledProcessError as e:
                print(f"\nError during command execution: {e}")
                if e.stderr:
                    print(f"Error details: {e.stderr}")
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
    elif confirmation == 'x':
        explain_command(command)
        # Ask again after showing explanation
        execute_command(command)
    else:
        print("\n Command cancelled")

# Add this function to your code
def test_microphone(input_device_index=None):
    """Test if the microphone is working and capturing audio"""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    RECORD_SECONDS = 10
    
    print("\n===== MICROPHONE TEST =====")
    print("Speaking into your microphone for 10 seconds...")
    print("You should see the bars move when you speak")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)
    
    # Show audio levels for 10 seconds
    max_volume = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Calculate audio level
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume_norm = np.linalg.norm(audio_data) / 32767
        max_volume = max(max_volume, volume_norm)
        
        bar_length = int(50 * volume_norm)
        print(f"\rLevel: [{'|' * bar_length}{' ' * (50 - bar_length)}] {volume_norm:.4f}", end="")
    
    print("\n")
    print(f"Maximum volume detected: {max_volume:.4f}")
    
    if max_volume < 0.01:
        print("No audio detected! Your microphone might not be working.")
        print("Try a different microphone or check your sound settings.")
    elif max_volume < 0.1:
        print(" Very low audio levels detected. Try speaking louder or adjusting your microphone.")
    else:
        print("Microphone is working correctly!")
    
    print("===== TEST COMPLETE =====\n")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

def load_plugins():
    """Loads plugins from the 'plugins' directory relative to the script."""
    # Construct the absolute path based on the script's location
    plugins_dir = os.path.join(SCRIPT_DIR, "plugins")

    loaded_plugins = {}
    # Use the absolute path for checking and creating
    if not os.path.exists(plugins_dir):
        # Optional: Print the full path for clarity during creation
        print(f"Plugins directory not found. Creating: {plugins_dir}")
        try:
            os.makedirs(plugins_dir) # Create the directory using the absolute path
        except OSError as e:
            print(f"Error creating plugins directory '{plugins_dir}': {e}")
            return loaded_plugins # Return empty if creation fails

    if not os.path.isdir(plugins_dir):
        print(f"Error: '{plugins_dir}' exists but is not a directory.")
        return loaded_plugins

    print(f"Loading plugins from: {plugins_dir}") # Optional: Confirm loading location
    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            plugin_path = os.path.join(plugins_dir, filename)
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Each plugin file must define a class named 'Plugin'
                if hasattr(module, "Plugin"):
                    plugin_class = getattr(module, "Plugin")
                    plugin_instance = plugin_class()
                    loaded_plugins[filename] = plugin_instance
                    print(f"Loaded plugin: {filename}")
                else:
                    print(f"No 'Plugin' class found in: {filename}")
    return loaded_plugins

def process_command_with_plugins(command, plugins):
    """
    Passes the command through each plugin's 'execute' method in sequence.
    """
    for plugin in plugins.values():
        command = plugin.execute(command)
    return command

def diagnose_microphones():
    """
    Provides detailed diagnostic information about available microphones
    """
    print("\n===== MICROPHONE DIAGNOSTIC =====")
    p = pyaudio.PyAudio()
    
    # Get host API info
    print("\nHost API Information:")
    for i in range(p.get_host_api_count()):
        info = p.get_host_api_info_by_index(i)
        print(f"API {i}: {info['name']}, devices: {info['deviceCount']}")
    
    # Get device info
    print("\nMicrophone Details:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        
        # Check if this is an input device
        if info['maxInputChannels'] > 0:
            print(f"\nDevice {i}: {info['name']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']} Hz")
            print(f"  Input Channels: {info['maxInputChannels']}")
            print(f"  Default Input: {'Yes' if p.get_default_input_device_info()['index'] == i else 'No'}")
            
            # Try to open the device to check if it's really available
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=4096,
                    start=False  # Don't actually start streaming
                )
                print("  Status: Available ✅")
                stream.close()
            except Exception as e:
                print(f"  Status: ERROR - {str(e)} ❌")
    
    p.terminate()
    print("\n===== END DIAGNOSTIC =====")

# Add this new function to show history
def show_command_history():
    """Display command history and allow reuse."""
    if not command_history:
        print("\n No command history available.")
        return
        
    print("\n Command History:")
    for i, cmd in enumerate(command_history):
        # Truncate long commands for display
        display_cmd = cmd if len(cmd) < 60 else cmd[:57] + "..."
        print(f" {i+1}. {display_cmd}")
        
    try:
        choice = input("\n Select a command to reuse (number) or (C)ancel: ").strip().lower()
        if choice == 'c':
            return None
            
        idx = int(choice) - 1
        if 0 <= idx < len(command_history):
            return command_history[idx]
        else:
            print(" Invalid selection.")
            return None
    except ValueError:
        print(" Invalid input.")
        return None

# Add this function
def load_config():
    """Load application configuration."""
    config_path = os.path.join(SCRIPT_DIR, "config.ini")
    config = configparser.ConfigParser()
    
    # Create default config if not exists
    if not os.path.exists(config_path):
        print(f"Creating new configuration file at {config_path}")
        
        # Check if example config exists to use as template
        example_config_path = os.path.join(SCRIPT_DIR, "config.ini.example")
        if os.path.exists(example_config_path):
            config.read(example_config_path)
            
            # Don't copy API key from example
            if 'API' in config and 'mistral_api_key' in config['API']:
                config['API']['mistral_api_key'] = ''
                
            # Save as new config
            with open(config_path, 'w') as f:
                config.write(f)
                
            print(f"Created configuration from template file.")
        else:
            # Set default values
            config['API'] = {
                'mistral_api_key': '',
                'mistral_endpoint': 'https://api.mistral.ai/v1/chat/completions',
                'mistral_model': 'mistral-tiny'
            }
            config['Whisper'] = {
                'model_size': 'small',
                'device': 'cpu'
            }
            config['Paths'] = {
                'ffmpeg_path': '',
                'plugins_dir': os.path.join(SCRIPT_DIR, 'plugins')
            }
            config['Recording'] = {
                'max_seconds': '12',
                'silence_threshold': '100',
                'silence_duration': '3.0',
                'rate': '16000',
                'chunk_size': '4096'
            }
            config['History'] = {
                'max_commands': '20'
            }
            
            with open(config_path, 'w') as f:
                config.write(f)
                
            print(f"Created default configuration.")
    
    # Now load the existing config
    config.read(config_path)
    
    # Check for environment variables
    mistral_key_env = os.environ.get('MISTRAL_API_KEY')
    if mistral_key_env and not config.get('API', 'mistral_api_key', fallback=''):
        config['API']['mistral_api_key'] = mistral_key_env
        print("Using Mistral API key from environment variable")
    
    return config

# Add this function
def manage_tasks():
    """Manage saved tasks (sequences of commands)."""
    tasks_file = os.path.join(SCRIPT_DIR, "tasks.json")
    tasks = {}
    
    # Load existing tasks
    if os.path.exists(tasks_file):
        try:
            with open(tasks_file, 'r') as f:
                tasks = json.load(f)
        except:
            print(" Error loading tasks file.")
    
    while True:
        print("\n Task Management:")
        print(" 1. List all tasks")
        print(" 2. Create new task")
        print(" 3. Run a task")
        print(" 4. Delete a task")
        print(" 5. Return to main menu")
        
        choice = input("\n Choose an option (1-5): ").strip()
        
        if choice == "1":
            if not tasks:
                print(" No saved tasks.")
                continue
                
            print("\n Saved Tasks:")
            for name in tasks:
                cmd_count = len(tasks[name])
                print(f" - {name} ({cmd_count} commands)")
                
        elif choice == "2":
            name = input(" Enter a name for the new task: ").strip()
            if not name:
                print(" Task name cannot be empty.")
                continue
                
            if name in tasks:
                overwrite = input(f" Task '{name}' already exists. Overwrite? (y/n): ").lower()
                if overwrite != 'y':
                    continue
            
            commands = []
            print(" Enter commands one by one (enter blank line to finish):")
            while True:
                cmd = input(" > ").strip()
                if not cmd:
                    break
                commands.append(cmd)
            
            if commands:
                tasks[name] = commands
                with open(tasks_file, 'w') as f:
                    json.dump(tasks, f, indent=2)
                print(f" Task '{name}' saved with {len(commands)} commands.")
            else:
                print(" No commands added. Task creation cancelled.")
                
        elif choice == "3":
            if not tasks:
                print(" No saved tasks.")
                continue
                
            print("\n Available Tasks:")
            task_names = list(tasks.keys())
            for i, name in enumerate(task_names, 1):
                print(f" {i}. {name}")
                
            try:
                idx = int(input("\n Select task to run (number): ")) - 1
                if 0 <= idx < len(task_names):
                    task_name = task_names[idx]
                    commands = tasks[task_name]
                    
                    print(f"\n Running task '{task_name}' ({len(commands)} commands):")
                    for i, cmd in enumerate(commands, 1):
                        print(f"\n Command {i}/{len(commands)}: {cmd}")
                        confirm = input(" Run this command? (Y/N/Skip rest): ").lower()
                        
                        if confirm == 'y':
                            execute_command(cmd)
                        elif confirm == 'skip rest':
                            break
                else:
                    print(" Invalid selection.")
            except ValueError:
                print(" Invalid input.")
                
        elif choice == "4":
            if not tasks:
                print(" No saved tasks.")
                continue
                
            print("\n Available Tasks:")
            task_names = list(tasks.keys())
            for i, name in enumerate(task_names, 1):
                print(f" {i}. {name}")
                
            try:
                idx = int(input("\n Select task to delete (number): ")) - 1
                if 0 <= idx < len(task_names):
                    task_name = task_names[idx]
                    confirm = input(f" Are you sure you want to delete '{task_name}'? (y/n): ").lower()
                    
                    if confirm == 'y':
                        del tasks[task_name]
                        with open(tasks_file, 'w') as f:
                            json.dump(tasks, f, indent=2)
                        print(f" Task '{task_name}' deleted.")
                else:
                    print(" Invalid selection.")
            except ValueError:
                print(" Invalid input.")
                
        elif choice == "5":
            break
        1
def import_wav_tester():
    """Import the simple WAV tester module if available"""
    tester_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav_file_tester.py")
    
    if not os.path.exists(tester_path):
        return None
        
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wav__file_tester", tester_path)
        if spec and spec.loader:
            tester = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tester)
            return tester
        return None
    except Exception as e:
        print(f"Error importing WAV tester: {e}")
        return None

# function for the batch tester    
def import_batch_tester():
    """Import the batch WAV tester module"""
    tester_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_tester.py")
    
    if not os.path.exists(tester_path):
        return None
        
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("batch_tester.py", tester_path)
        if spec and spec.loader:
            tester = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tester)
            return tester
        return None
    except Exception as e:
        print(f"Error importing batch WAV tester: {e}")
        return None

# Modify main function to check OS at the start
def main():
    config = load_config()
    
    # Initialize metrics file
    metrics_file = initialize_metrics()
    print(f"Metrics will be saved to: {metrics_file}")
    
    # Now we'll use the config throughout instead of global variables
    
    # Set ffmpeg path if specified
    ffmpeg_path = config.get('Paths', 'ffmpeg_path', fallback='')
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + ffmpeg_path
        
    # Update whisper model loading to use config
    whisper_model_size = config.get('Whisper', 'model_size', fallback='small')
    # ...rest of main()

    # Load the Whisper model at the start of main
    global whisper_model
    if whisper_model is None:
        try:
            print(f"Loading Whisper '{whisper_model_size}' model (more accurate, may take longer)...")
            whisper_model = whisper.load_model(whisper_model_size)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Voice input will be disabled.")
            # Continue execution without voice capability

    # Detect shell first
    shell = detect_shell()

    # Exit if OS is not supported
    if shell == "unsupported":
        print("Error: This application only runs on Windows.")
        sys.exit(1) # Exit the script

    # Continue with normal startup if on Windows
    print(f"\n✨ Voice Command CLI - Using {shell.upper()} shell")
    
    # 1) Load plugins
    loaded_plugins = load_plugins()

    # 2) Ask the user which mic to use
    mic_index = select_microphone() # Returns None if skipped/invalid

    # 3) Main menu loop
    while True:
        print("\n Choose an option:")
        print(" (T) Type command")
        if mic_index is not None:
            print(" (V) Voice command")
            print(" (M) Test microphone")
        print(" (B) Batch test WAV files")    
        print(" (D) Microphone diagnostic")
        print(" (H) Command history")
        print(" (S) Saved tasks")
        print(" (C) Command chain")
        print(" (M) Metrics analysis")
        print(" (W) Test WAV file")
        print(" (Q) Quit")

        prompt_options = "B/T/D/H/S/C/M/W/Q"
        if mic_index is not None:
            prompt_options = "B/T/V/M/D/H/S/C/M/W/Q"

        choice = input(f"\nEnter choice ({prompt_options}): ").strip().lower()

        if choice.lower() == "t":
            user_input = input("Enter command: ").strip().lower()
            plugin_output = process_command_with_plugins(user_input, loaded_plugins)
            # Pass the 'shell' variable when calling
            command_options = process_with_mistral(plugin_output, shell)

            if command_options and isinstance(command_options, dict) and command_options.get("is_chain"):
                # This is a command chain, handle it differently
                execute_command_chain(command_options)
            elif command_options:  # This is the regular single-command case
                # Display options to user - IMPROVED FORMATTING
                print("\n Command Options:")
                for i, option in enumerate(command_options, 1):
                    cmd_text = option.get('command', 'N/A')
                    explanation_text = option.get('explanation', 'No explanation provided.')
                    # Print command and explanation on separate lines with indentation
                    print(f" {i}. Command:     {cmd_text}")
                    print(f"    Explanation: {explanation_text}\n")  # Add newline for spacing
                
                # Let user choose
                try:
                    select_choice = input(f"\nSelect option (1-{len(command_options)}) or (C)ancel: ").strip() # Use different variable name
                    if select_choice.lower() == 'c':
                        print(" Command cancelled")
                        continue

                    option_num = int(select_choice) - 1
                    if 0 <= option_num < len(command_options):
                        selected_command = command_options[option_num].get('command')
                        if selected_command:
                             execute_command(selected_command)
                        else:
                             print(" Error: Selected option has no command.")
                    else:
                        print(" Invalid selection")
                except ValueError:
                    print(" Invalid input, command cancelled")
            else:
                print(" Could not get command suggestions from Mistral.")

        #import batch tester
        elif choice == "b":
          wav_tester = import_batch_tester()
    
          if wav_tester is None:
              print("Batch WAV tester module not found. Make sure streamlined_wav_batch_tester.py is in the same directory.")
              continue
            
          test_dir = input("\nEnter directory with WAV files [./resource]: ").strip()
          if not test_dir:
              test_dir = "./resource"
          
          # Pass the currently loaded resources to avoid reimporting them
          wav_tester.batch_test_wav_files(
              directory_path=test_dir,
              whisper_model=whisper_model,
              plugins=loaded_plugins,
              shell=shell
          )

        # --- Voice and Mic Test options ---
        # Only process 'v' and 'm' if mic_index is valid
        elif choice.lower() == "v" and mic_index is not None:
            command_text, metrics = get_voice_command(input_device_index=mic_index)
            if command_text:
                plugin_output = process_command_with_plugins(command_text, loaded_plugins)
                # Pass the 'shell' variable when calling
                command_options = process_with_mistral(plugin_output, shell)

                if command_options and isinstance(command_options, dict) and command_options.get("is_chain"):
                    # This is a command chain, handle it differently
                    execute_command_chain(command_options)
                elif command_options:  # This is the regular single-command case
                    # Display options to user - IMPROVED FORMATTING
                    print("\n Command Options:")
                    for i, option in enumerate(command_options, 1):
                        cmd_text = option.get('command', 'N/A')
                        explanation_text = option.get('explanation', 'No explanation provided.')
                        # Print command and explanation on separate lines with indentation
                        print(f" {i}. Command:     {cmd_text}")
                        print(f"    Explanation: {explanation_text}\n")  # Add newline for spacing
                    
                    # Let user choose
                    try:
                        select_choice = input(f"\nSelect option (1-{len(command_options)}) or (C)ancel: ").strip() # Use different variable name
                        if select_choice.lower() == 'c':
                            print(" Command cancelled")
                            continue

                        option_num = int(select_choice) - 1
                        if 0 <= option_num < len(command_options):
                            selected_command = command_options[option_num].get('command')
                            if selected_command:
                                execute_command(selected_command)
                            else:
                                print(" Error: Selected option has no command.")
                        else:
                            print(" Invalid selection")
                    except ValueError:
                        print(" Invalid input, command cancelled")
                else:
                    print(" Could not get command suggestions from Mistral.")
            continue

        elif choice.lower() == "m" and mic_index is not None:
            test_microphone(input_device_index=mic_index)
        # --- End of Voice/Mic options ---

        elif choice.lower() == "d":
            diagnose_microphones()

        elif choice.lower() == "h":
            selected_command = show_command_history()
            if selected_command:
                execute_command(selected_command)

        elif choice.lower() == "s":
            manage_tasks()

        elif choice.lower() == "c":
            user_input = input("Enter complex command (e.g., 'create a file and move it'): ").strip()
            debug_mode = input("Enable debug mode? (y/n): ").strip().lower() == 'y'
            plugin_output = process_command_with_plugins(user_input, loaded_plugins)
            # Force chain mode and optionally enable debug
            command_options = process_with_mistral(plugin_output, shell, force_chain=True, debug=debug_mode)
            
            if command_options and isinstance(command_options, dict) and command_options.get("is_chain"):
                execute_command_chain(command_options)
            else:
                print(" Could not create a command chain. Try being more specific.")
                if command_options and not isinstance(command_options, dict):
                    # Show single command options if available
                    print(" However, I found these single command options:")
                    for i, option in enumerate(command_options, 1):
                        cmd = option.get('command', 'N/A')
                        explanation = option.get('explanation', '')
                        print(f" {i}. {cmd}\n    {explanation}")

        elif choice.lower() == "m":
            display_metrics()
        
        elif choice.lower() == "w":
          #import the WAV tester module
          wav_tester = import_wav_tester()
    
        if wav_tester is None:
          print("WAV tester module not found. Make sure wav_tester_file.py is in the same directory.")
          continue
        
        # Get WAV file selection
        wav_file = wav_tester.select_wav_file()
    
        if wav_file:
          #Test the file
          transcription = wav_tester.test_wav_file(wav_file, whisper_model)
        
          if transcription:
            # Ask if user wants to use as command
            process = input("\nProcess this as a command? (y/n): ").strip().lower()
            
            if process == 'y':
              # Process just like voice input
              plugin_output = process_command_with_plugins(transcription, loaded_plugins)
              command_options = process_with_mistral(plugin_output, shell)
                
              if command_options and isinstance(command_options, dict) and command_options.get("is_chain"):
                # Handle command chain
                execute_command_chain(command_options)
              elif command_options:
                # Handle normal command options
                print("\nCommand Options:")
                for i, option in enumerate(command_options, 1):
                  cmd_text = option.get('command', 'N/A')
                  explanation = option.get('explanation', '')
                  print(f"{i}. {cmd_text}")
                  print(f"   {explanation}")
                    
                    # Let user select
                try:
                  choice = input("\nSelect option (1-n) or C to cancel: ").strip().lower()
                  if choice != 'c':
                      idx = int(choice) - 1
                      if 0 <= idx < len(command_options):
                          selected_command = command_options[idx].get('command')
                          if selected_command:
                              execute_command(selected_command)
                except ValueError:
                  print("Invalid selection.")
              else:
                print("Could not process command.")

        elif choice == "q":
            print("Exiting program.")
            break
        else:
            # Handle invalid choices, considering the context
            if mic_index is None and choice in ['v', 'm']:
                 print("Invalid choice. Microphone not selected.")
            else:
                 print("Invalid choice. Please try again.")

    load_command_history()  # Add this line

# Simplify the main block
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def explain_command(command):
    """Use Mistral AI to explain what a command does in plain language."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are a helpful command-line expert. Explain commands in simple terms."},
            {"role": "user", "content": f"Please explain what this command does in simple terms: {command}"}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    print("\n Getting explanation from Mistral AI...")
    
    try:
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        explanation = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        print("\n Command Explanation:")
        print(f" {explanation}")
        
    except Exception as e:
        print(f" Error getting explanation: {e}")

def parse_mistral_response(response, original_request, is_chain_request=False):
    """Parse Mistral AI response, handling both single commands and command chains."""
    try:
        content = response["choices"][0]["message"]["content"].strip()
        
        if is_chain_request:
            # Enhanced parsing for command chains
            command_chain = []
            
            # Try multiple patterns for command detection
            step_pattern = r'Step\s+\d+:?\s*\n?COMMAND:\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=Step\s+\d+:|$)'
            numbered_pattern = r'\d+\.\s*COMMAND:\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=\d+\.\s*COMMAND:|\Z)'
            command_pattern = r'COMMAND:\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=COMMAND:|\Z)'
            
            # Try all patterns
            matches = re.findall(step_pattern, content, re.DOTALL) or \
                      re.findall(numbered_pattern, content, re.DOTALL) or \
                      re.findall(command_pattern, content, re.DOTALL)
            
            # No matches with COMMAND prefix, try alternative formats
            if not matches:
                # Look for numbered list with commands in backticks
                alt_pattern = r'\d+\.\s*.*?`([^`]+)`(.*?)(?=\d+\.\s*|\Z)'
                matches = re.findall(alt_pattern, content, re.DOTALL)
            
            # Still no matches, try extracting commands in backticks with explanations
            if not matches:
                backtick_pattern = r'`([^`]+)`(.*?)(?=`[^`]+`|\Z)'
                matches = re.findall(backtick_pattern, content, re.DOTALL)
            
            if matches:
                for cmd, explanation in matches:
                    command_chain.append({
                        "command": cmd.strip(),
                        "explanation": explanation.strip()
                    })
            
            # If we found a command chain, return it with a special flag
            if len(command_chain) > 1:  # Need at least 2 commands for a chain
                return {
                    "is_chain": True,
                    "steps": command_chain,
                    "original_request": original_request
                }
            elif len(command_chain) == 1:
                # If only one command found, return as a regular command
                return [command_chain[0]]
        
        # Default to regular parsing for single commands
        options = []
        command_pattern = r'`([^`]+)`\s*-?\s*(.+?)(?=\n\n|\n\d+\.|\Z)'
        matches = re.findall(command_pattern, content, re.DOTALL)
        
        for i, (cmd_text, explanation) in enumerate(matches, 1):
            options.append({
                'command': cmd_text.strip(),
                'explanation': explanation.strip()
            })
        
        return options
    except Exception as e:
        print(f" Error parsing Mistral response: {e}")
        return None

def execute_command_chain(command_chain):
    """Execute a sequence of commands."""
    if not command_chain or not isinstance(command_chain, dict) or not command_chain.get("is_chain"):
        print(" Error: Invalid command chain format")
        return
    
    steps = command_chain.get("steps", [])
    original_request = command_chain.get("original_request", "")
    
    if not steps:
        print(" Error: Command chain contains no steps")
        return
    
    print(f"\n 🔄 Command Chain for: \"{original_request}\"")
    print(f" Total steps: {len(steps)}")
    
    # Ask for confirmation to run the chain
    print("\n Review the command chain:")
    for i, step in enumerate(steps, 1):
        cmd = step.get("command", "")
        explanation = step.get("explanation", "")
        print(f"\n Step {i}: {cmd}")
        print(f" Explanation: {explanation}")
    
    print("\n⚠️ WARNING: This will execute multiple commands in sequence.")
    confirm = input(" Execute this command chain? (Y)es / (N)o / (S)tep-by-step: ").strip().lower()
    
    if confirm == 'n':
        print(" Command chain execution cancelled")
        return
    
    # Execute all commands in sequence
    for i, step in enumerate(steps, 1):
        cmd = step.get("command", "")
        print(f"\n Executing step {i}/{len(steps)}: {cmd}")
        
        # Check for dangerous commands
        if is_command_dangerous(cmd):
            print(" ⚠️ WARNING: This command looks potentially dangerous!")
            override = input(" Execute anyway? (y/n): ").strip().lower()
            if override != 'y':
                print(" Skipping this command")
                continue
        
        # For step-by-step execution, ask before each command
        if confirm == 's':
            step_confirm = input(" Execute this step? (Y)es / (N)o / (A)bort chain: ").strip().lower()
            if step_confirm == 'n':
                print(" Skipping this step")
                continue
            elif step_confirm == 'a':
                print(" Aborting command chain execution")
                break
        
        # Execute the command
        shell = detect_shell()
        try:
            print(" Executing command...")
            start_time = time.time()
            
            # Use the appropriate execution method based on shell
            if shell == "powershell":
                process = subprocess.Popen(["powershell", "-Command", cmd], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                          text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # cmd
                process = subprocess.Popen(["cmd", "/c", cmd], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            stdout, stderr = process.communicate()
            
            end_time = time.time()
            exec_time = end_time - start_time
            
            # Print output with line count for long outputs
            if stdout:
                lines = stdout.splitlines()
                print("\n" + "\n".join(lines[:50]))  # Show first 50 lines
                if len(lines) > 50:
                    print(f"\n... {len(lines) - 50} more lines (showing 50/{len(lines)})")
                print(f"\n Total: {len(lines)} lines of output")
            
            if stderr:
                print(f"\n Error output:\n{stderr}")
            
            print(f"\n Command step {i} execution complete (took {exec_time:.2f} seconds)")
            
            # Add to command history
            if cmd not in ["", None]:
                command_history.appendleft(cmd)
                save_command_history()
                
        except Exception as e:
            print(f"\n Error executing command: {e}")
            if confirm == 's':
                abort = input(" An error occurred. Abort chain? (y/n): ").strip().lower()
                if abort == 'y':
                    print(" Aborting command chain execution")
                    break
    
    print("\n ✅ Command chain execution complete")

# Add this function to create a metrics system
def initialize_metrics():
    """Initialize the metrics tracking system."""
    metrics_file = os.path.join(SCRIPT_DIR, "usage_metrics.csv")
    # Check if file exists, if not create with headers
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'input_mode', 'command_type', 'execution_time', 
                'was_successful', 'required_correction', 'transcription_accuracy',
                'suggestions_count', 'suggestion_selected', 'command_length'
            ])
    return metrics_file

def log_metrics(metrics_data):
    """Log command execution metrics to CSV file."""
    metrics_file = os.path.join(SCRIPT_DIR, "usage_metrics.csv")
    
    try:
        # Make sure metrics_data contains the required fields
        if 'input_mode' not in metrics_data:
            metrics_data['input_mode'] = 'text'  # Default value
        
        # Ensure all keys exist to prevent KeyError
        for key in ['command_type', 'execution_time', 'was_successful', 'required_correction',
                   'transcription_accuracy', 'suggestions_count', 'suggestion_selected', 'command_length']:
            if key not in metrics_data:
                metrics_data[key] = ''
        
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Add timestamp to the data
            metrics_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Convert dict to list in the correct order
            row = [
                metrics_data.get('timestamp', ''),
                metrics_data.get('input_mode', 'text'),  # Default to 'text' if missing
                metrics_data.get('command_type', 'single'),  # Default to 'single' if missing
                metrics_data.get('execution_time', 0),
                metrics_data.get('was_successful', False),
                metrics_data.get('required_correction', False),
                metrics_data.get('transcription_accuracy', 0),
                metrics_data.get('suggestions_count', 0),
                metrics_data.get('suggestion_selected', 0),
                metrics_data.get('command_length', 0)
            ]
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error logging metrics: {e}")
        return False

def display_metrics():
    """Display and analyze usage metrics."""
    metrics_file = os.path.join(SCRIPT_DIR, "usage_metrics.csv")
    
    if not os.path.exists(metrics_file):
        print("\n No metrics data available yet.")
        return
    
    try:
        # Read metrics data
        metrics = []
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip the header row
            header = next(reader, None)
            
            for row in reader:
                if len(row) >= 10:  # Ensure we have at least the expected number of columns
                    metrics.append({
                        'timestamp': row[0],
                        'input_mode': row[1],
                        'command_type': row[2],
                        'execution_time': row[3],
                        'was_successful': row[4],
                        'required_correction': row[5],
                        'transcription_accuracy': row[6],
                        'suggestions_count': row[7],
                        'suggestion_selected': row[8],
                        'command_length': row[9]
                    })
        
        if not metrics:
            print("\n No metrics data available yet.")
            return
        
        # Calculate statistics
        total_commands = len(metrics)
        voice_commands = sum(1 for m in metrics if m['input_mode'] == 'voice')
        text_commands = sum(1 for m in metrics if m['input_mode'] == 'text')
        
        # Convert to float before doing math
        try:
            success_rate = sum(1 for m in metrics if m['was_successful'] == 'True') / total_commands
        except:
            success_rate = 0
            
        try:
            avg_exec_time = sum(float(m['execution_time']) for m in metrics) / total_commands
        except:
            avg_exec_time = 0
            
        try:
            correction_rate = sum(1 for m in metrics if m['required_correction'] == 'True') / total_commands
        except:
            correction_rate = 0
        
        # Display summary
        print("\n📊 Usage Metrics Summary:")
        print(f" Total commands executed: {total_commands}")
        print(f" Voice commands: {voice_commands} ({voice_commands/total_commands*100 if total_commands else 0:.1f}%)")
        print(f" Text commands: {text_commands} ({text_commands/total_commands*100 if total_commands else 0:.1f}%)")
        print(f" Success rate: {success_rate*100:.1f}%")
        print(f" Average execution time: {avg_exec_time:.2f} seconds")
        print(f" Commands requiring correction: {correction_rate*100:.1f}%")
        
        # Ask if user wants detailed report
        choice = input("\n Do you want to (V)iew detailed metrics or (E)xport to CSV? (V/E/C)ancel: ").lower()
        
        if choice == 'v':
            # Show more detailed metrics, perhaps the last 10 commands
            print("\n Last 10 commands:")
            for i, m in enumerate(metrics[-10:], 1):
                try:
                    exec_time = float(m['execution_time'])
                except:
                    exec_time = 0
                    
                print(f" {i}. [{m['timestamp']}] {m['input_mode']} command, " +
                      f"Execution: {exec_time:.2f}s, " +
                      f"Success: {m['was_successful']}")
        
        elif choice == 'e':
            # Export to a more readable format
            export_file = os.path.join(SCRIPT_DIR, "metrics_report.csv")
            with open(export_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Command Type', 'Success', 'Execution Time'])
                for m in metrics:
                    writer.writerow([
                        m['timestamp'], 
                        f"{m['input_mode']} - {m['command_type']}", 
                        m['was_successful'], 
                        m['execution_time']
                    ])
            print(f"\n Metrics exported to {export_file}")
    
    except Exception as e:
        print(f"\n Error analyzing metrics: {str(e)}")
        # Add traceback for debugging
        import traceback
        print(traceback.format_exc())

def initialize_command_history():
    """Initialize the command history with configurable size."""
    config = load_config()
    max_history = config.getint('History', 'max_commands', fallback=20)
    
    global command_history
    command_history = deque(maxlen=max_history)
    load_command_history()
