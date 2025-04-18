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

# Get the absolute path to the directory containing THIS script file
# This works correctly even when run via the voice.bat wrapper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use a fixed API key instead of environment variables
MISTRAL_API_KEY = "QnypDzjhUzYGMUKVTyqJrpngORGi15TG"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Add this near the top of your file, after imports
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + r"C:\Users\Earmy\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"

# Define a blacklist of dangerous command patterns (using regex)
# This list can be expanded. Covers common deletion, formatting, execution risks.
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
            # Optional: Print which pattern matched for debugging
            # print(f"Potential danger detected: Command '{command}' matched pattern '{pattern.pattern}'")
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
        # Basic validation: check if index is within the range of devices
        # Note: This doesn't guarantee it's a valid *input* device,
        # but it's a basic sanity check.
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

# Replace with just this global variable declaration:
model = None  # Will be initialized in __main__

def get_voice_command(input_device_index=None):
    """Records audio and transcribes it using Whisper."""
    global model
    if model is None:
        print("Error: Whisper model not loaded.")
        return None

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    RECORD_SECONDS = 12  # Increased to 12 seconds for more time
    SILENCE_THRESHOLD = 100  # Reduced threshold - more sensitive to quieter sounds
    SILENCE_DURATION = 3.0   # Longer silence duration before stopping

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
        result = model.transcribe(tmp_audio_path, language="en", fp16=False) # fp16=False for CPU
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

    return transcribed_text

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

# Update get_system_prompt function
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
            "Keep the commands focused on displaying output in the terminal by default."
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
            "Keep the commands focused on displaying output in the terminal by default."
        )
    else: # Handle unsupported case
        return "This operating system is not supported. This tool only works on Windows."

# Modify process_with_mistral to filter results
def process_with_mistral(command, shell):
    """
    Sends command text to Mistral AI, gets suggestions, and filters dangerous ones.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    # Build prompt to get better structured output
    system_prompt = (
        "You are a helpful assistant that generates shell commands for PowerShell or CMD. "
        "The user will describe what they want to do, and you will provide the exact commands they should run. "
        "Provide 1-3 command options with explanations. "
        "Format your response as a JSON array of objects with 'command' and 'explanation' fields. "
        "Example format: ```json [{\"command\": \"dir\", \"explanation\": \"Lists files in current directory\"}] ```"
    )
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {shell.upper()} commands for: '{command}'"}
        ],
        "temperature": 0.3,  # Lower temperature for more focused output
        "max_tokens": 800
    }
    
    print(f" Consulting Mistral AI for {shell.upper()} command suggestions...")
    
    start_time = time.time()
    try:
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if 'choices' not in response_json or not response_json['choices']:
            print("Error: Unexpected response structure from Mistral.")
            return None
        content = response_json['choices'][0]['message']['content']

        try:
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content)
            if json_match:
                json_str = json_match.group(1).strip()
                commands = json.loads(json_str)
            else:
                commands = json.loads(content) # Fallback

            if not (isinstance(commands, list) and all(isinstance(cmd, dict) for cmd in commands)):
                 print(f"Extracted data is not a list of dictionaries: {type(commands)}")
                 return None

            # --- FILTER DANGEROUS COMMANDS ---
            safe_commands = []
            filtered_count = 0
            filtered_examples = []  # Keep track of what was filtered for feedback

            for cmd_option in commands:
                cmd_text = cmd_option.get('command')
                if cmd_text and not is_command_dangerous(cmd_text):
                    safe_commands.append(cmd_option)
                else:
                    filtered_count += 1
                    # Store a sanitized version of the filtered command for user awareness
                    # Only store the first two as examples
                    if len(filtered_examples) < 2 and cmd_text:
                        # Anonymize the command slightly for security
                        cmd_preview = cmd_text[:10] + "..." if len(cmd_text) > 10 else cmd_text
                        filtered_examples.append(cmd_preview)

            if filtered_count > 0:
                print(f"\n🛡️ Security: Filtered {filtered_count} potentially dangerous command(s).")
                if filtered_examples:
                    print(f"   Example(s): {', '.join(filtered_examples)}")

            if not safe_commands:
                print("No safe command suggestions were returned after filtering.")
                return None

            return safe_commands # Return only the safe commands
            # --- END FILTER ---

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None

    except Exception as e:
        print(f"Error during Mistral API request: {e}")
        return None
    finally:
        pass # Keep timing removed

# Modify execute_command confirmation for clarity
def execute_command(command):
    shell = detect_shell()
    if shell == "unsupported":
        print("\nError: Cannot execute command. This tool only supports Windows (CMD/PowerShell).")
        return

    print(f"\n Command Ready to Execute ({shell.upper()}):")
    # Display command clearly, maybe wrap long ones
    if len(command) > 80:
         print(f"  {command[:77]}...")
         # Consider showing the full command if needed, or just the start
    else:
         print(f"  {command}")

    # Emphasize checking the command
    print("\n⚠️ IMPORTANT: Review the command carefully before executing! ⚠️")
    confirmation = input("Execute this command? (Y)es / (N)o / (E)dit: ").strip().lower()

    if confirmation == 'y':
        print("\n Executing command...\n")
        try:
            if shell == "powershell":
                # Use shell=True for robustness, especially with complex commands or paths
                subprocess.run(["powershell", "-Command", command], check=True, shell=True)
            elif shell == "cmd":
                # Use shell=True for CMD execution
                subprocess.run(command, check=True, shell=True)
            print("\n Command execution complete")
        except subprocess.CalledProcessError as e:
            print(f"\nError during command execution: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
    elif confirmation == 'e':
        edited = input("\n Enter modified command: ").strip()
        if edited:
            print("\n Executing modified command...\n")
            try:
                if shell == "powershell":
                    subprocess.run(["powershell", "-Command", edited], check=True, shell=True)
                elif shell == "cmd":
                    subprocess.run(edited, check=True, shell=True)
                print("\n Command execution complete")
            except subprocess.CalledProcessError as e:
                print(f"\nError during command execution: {e}")
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
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

# Modify main function to check OS at the start
def main():
    # Load the Whisper model at the start of main
    global model
    if model is None:
        try:
            print("Loading Whisper 'small' model (more accurate, may take longer)...")
            model = whisper.load_model("small")
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
        # Only show Voice and Mic Test options if a mic was selected
        if mic_index is not None:
            print(" (V) Voice command")
            print(" (M) Test microphone")
        print(" (D) Microphone diagnostic")
        print(" (Q) Quit")

        prompt_options = "T/D/Q"
        if mic_index is not None:
            prompt_options = "T/V/M/D/Q"

        choice = input(f"\nEnter choice ({prompt_options}): ").strip().lower()

        if choice == "t":
            user_input = input("Enter command: ").strip().lower()
            plugin_output = process_command_with_plugins(user_input, loaded_plugins)
            # Pass the 'shell' variable when calling
            command_options = process_with_mistral(plugin_output, shell)

            if command_options: # Check if we got options
                 # Display options to user - IMPROVED FORMATTING
                print("\n Command Options:")
                for i, option in enumerate(command_options, 1):
                    cmd_text = option.get('command', 'N/A')
                    explanation_text = option.get('explanation', 'No explanation provided.')
                    # Print command and explanation on separate lines with indentation
                    print(f" {i}. Command:     {cmd_text}")
                    print(f"    Explanation: {explanation_text}\n") # Add a newline for spacing

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

        # --- Voice and Mic Test options ---
        # Only process 'v' and 'm' if mic_index is valid
        elif choice == "v" and mic_index is not None:
            command_text = get_voice_command(input_device_index=mic_index)
            if command_text:
                plugin_output = process_command_with_plugins(command_text, loaded_plugins)
                # Pass the 'shell' variable when calling
                command_options = process_with_mistral(plugin_output, shell)

                if command_options: # Check if we got options
                    # Display options to user - IMPROVED FORMATTING
                    print("\n Command Options:")
                    for i, option in enumerate(command_options, 1):
                        cmd_text = option.get('command', 'N/A')
                        explanation_text = option.get('explanation', 'No explanation provided.')
                        # Print command and explanation on separate lines with indentation
                        print(f" {i}. Command:     {cmd_text}")
                        print(f"    Explanation: {explanation_text}\n") # Add a newline for spacing

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

        elif choice == "m" and mic_index is not None:
            test_microphone(input_device_index=mic_index)
        # --- End of Voice/Mic options ---

        elif choice == "d":
            diagnose_microphones()

        elif choice == "q":
            print("Exiting program.")
            break
        else:
            # Handle invalid choices, considering the context
            if mic_index is None and choice in ['v', 'm']:
                 print("Invalid choice. Microphone not selected.")
            else:
                 print("Invalid choice. Please try again.")

# Simplify the main block
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
