import gettext
import subprocess
import importlib.util
import os
import pyttsx3
import pyaudio
import time
import numpy as np
import tempfile
import wave
import sys
import contextlib
import json
import threading
import whisper
import requests
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a fixed API key instead of environment variables
MISTRAL_API_KEY = "QnypDzjhUzYGMUKVTyqJrpngORGi15TG"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Add this near the top of your file, after imports
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + r"C:\Users\Earmy\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"

# Setup gettext for multilingual support
def setup_localization():
    languages = ['en', 'es']  # Example languages: English and Spanish
    lang = gettext.translation('messages', localedir='locales', languages=languages, fallback=True)
    lang.install()
    global _
    _ = lang.gettext

setup_localization()

# Example usage in your existing functions
def select_microphone():
    mic = pyaudio.PyAudio()
    print(_("\nAvailable Microphones:"))
    info = mic.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount', 0)

    for i in range(0, numdevices):
        max_input_channels = mic.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels', 0)
        # Ensure max_input_channels is an integer and not None
        if max_input_channels is None:
            max_input_channels = 0
        else:
            try:
                max_input_channels = int(max_input_channels)
            except ValueError:
                max_input_channels = 0

        if max_input_channels > 0:
            print("Input Device id ", i, " - ", mic.get_device_info_by_host_api_device_index(0, i).get('name'))

    dev_index = input(_("Enter the device index for your microphone: "))
    try:
        dev_index = int(dev_index)
    except ValueError:
        print(_("Invalid input, using default device."))
        dev_index = 0
    return dev_index

def confirm_and_execute_command(transcribed_text):
    print(f"You said: {transcribed_text}")
    confirmation = input(_("Do you want to execute this command? (yes/edit/no): ")).strip().lower()
    if confirmation == 'edit':
        transcribed_text = input(_("Enter the corrected command: "))
    elif confirmation != 'yes':
        print(_("Command execution canceled."))
        return
    subprocess.run(transcribed_text, shell=True)

class PluginInterface:
    """
    An optional base interface (not strictly required).
    You can also define 'execute' directly in your plugin files.
    """
    def execute(self, command: str) -> str:
        """
        Execute a command using the plugin.
        This method should be overridden by subclasses to provide specific plugin functionality.
        """
        return command

def load_plugins():
    """
    Dynamically loads all .py files in the 'plugins' directory.
    Each plugin must define a class named 'Plugin' with a method 'execute'.
    """
    plugins_dir = "plugins"
    loaded_plugins = []

    if not os.path.isdir(plugins_dir):
        print(f"No '{plugins_dir}' folder found. Creating it now.")
        os.makedirs(plugins_dir)
        return loaded_plugins

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and not filename.startswith('__'):
            path_to_file = os.path.join(plugins_dir, filename)
            spec = importlib.util.spec_from_file_location("plugin_module", path_to_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Each plugin file must define a class named 'Plugin'
                if hasattr(module, "Plugin"):
                    plugin_class = getattr(module, "Plugin")
                    plugin_instance = plugin_class()
                    loaded_plugins.append(plugin_instance)
                    print(f"Loaded plugin: {filename}")
                else:
                    print(f"No 'Plugin' class found in: {filename}")
    return loaded_plugins

def process_command_with_plugins(command, plugins):
    """
    Passes the command through each plugin's 'execute' method in sequence.
    """
    for plugin in plugins:
        command = plugin.execute(command)
    return command

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Example usage
def get_voice_command(input_device_index=None):
    # Validate and convert input_device_index to an integer
    try:
        input_device_index = int(input_device_index) if input_device_index is not None else None
    except ValueError:
        print("Invalid input device index. Using default device.")
        input_device_index = None

    # Record audio
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    RECORD_SECONDS = 5  # Increased from 3 to 5 seconds
    
    print("🎤 Listening for a command (speak now)...")
    start_time = time.time()
    
    audio = pyaudio.PyAudio()
    
    # Print audio levels to help debug
    print("Speak into your microphone to test audio levels...")
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=input_device_index,  # Use the validated device index
                        frames_per_buffer=CHUNK)
    
    # Show a visual indicator while recording
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate audio level for visual feedback
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume_norm = np.linalg.norm(audio_data) / 32767
        bar_length = int(50 * volume_norm)
        print(f"\rRecording: [{'|' * bar_length}{' ' * (50 - bar_length)}] {i+1}/{int(RATE / CHUNK * RECORD_SECONDS)}", end="")
    
    print("\nProcessing audio...")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save recording to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_filename = f.name
    
    wf = wave.open(temp_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Transcribe with optimized settings
    result = model.transcribe(
        temp_filename,
        fp16=False,
        language="en",
        task="transcribe",
        without_timestamps=True
    )
    
    if result["text"]:  
        text = result["text"][0].strip()
    else:
        text = ""
    os.unlink(temp_filename)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if text:
        print(f"You said: {text}")
        print(f"Processing time: {processing_time:.2f} seconds")
        return text.lower()
    else:
        print("Could not understand audio")
        return None

def get_system_prompt():
    return (
        "You are a command translator for the Windows terminal. "
        "When asked to convert a natural language request to CMD, provide 3 different valid commands that accomplish the task, "
        "ranging from simple to advanced. Format your response as follows:\n\n"
        "1. Simple Command\n"
        "    `dir` - List all files in the current directory\n\n"
        "2. Intermediate Command\n"
        "    `dir /a` - List all files including hidden ones\n\n"
        "3. Advanced Command\n"
        "    `dir /a /s` - List all files recursively including hidden ones\n\n"
        "Always include the actual command in backticks (`) within your explanation. "
        "Ensure all commands are valid CMD syntax. Do not use redirection symbols (like > or |) unless specifically requested. "
        "Keep the commands focused on displaying output in the terminal by default."
    )

def process_with_mistral(command):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"Convert to CMD: {command}. Provide 3 different ways to accomplish this task, from simplest to most advanced."}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    print("\n Consulting Mistral AI for command suggestions...")
    start_time = time.time()

    try:
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()

        end_time = time.time()
        processing_time = end_time - start_time
        print(f" Received suggestions in {processing_time:.2f} seconds")

        return parse_mistral_response(response.json(), command)
    except requests.RequestException as e:
        print(f" Error calling Mistral: {e}")
        return None

def parse_mistral_response(response, original_request):
    content = response["choices"][0]["message"]["content"].strip()

    # Extract numbered commands with regex
    command_matches = re.findall(
        r'(\d+)\.\s+(.*?Command)\s*\n\s*📝\s*(.*?)(?=\n\d+\.|\n\n|$)',
        content,
        re.DOTALL
    )

    if not command_matches:
        # Try alternative pattern
        command_matches = re.findall(
            r'(\d+)\.\s+([^-\n]+)(?:\s*-\s*|\n\s*📝\s*)([^\n]+)',
            content
        )

    if not command_matches:
        # Fallback if pattern not found
        command_match = re.search(r'`([^`]+)`', content)
        if command_match:
            corrected_command = command_match.group(1).strip()
        else:
            corrected_command = content.split('\n')[0].strip()

        if not re.match(r'^[a-zA-Z]', corrected_command):
            print(" Mistral returned something that doesn't look like a valid command.")
            return None

        print(f" Mistral Suggestion: {corrected_command}")
        return corrected_command

    print("\n Command Suggestions for: " + original_request)
    print("─" * 60)

    commands = []
    for idx, (num, cmd_text, explanation) in enumerate(command_matches):
        # Extract the actual command
        if cmd_text.lower().endswith("command"):
            cmd_match = re.search(r'`([^`]+)`', explanation)
            if cmd_match:
                cmd = cmd_match.group(1).strip()
            else:
                cmd_words = explanation.split()
                cmd = next((word for word in cmd_words if re.match(r'^[a-zA-Z]', word)), "dir")
        else:
            cmd = cmd_text.strip()

        if not cmd or not re.match(r'^[a-zA-Z]', cmd):
            continue

        commands.append(cmd)
        print(f"\n{num}. {cmd}")
        print("   " + explanation.strip())

    print("\n" + "─" * 60)

    # Let user choose which command to execute
    if len(commands) > 1:
        while True:
            choice = input(f"\nSelect a command (1-{len(commands)}) or (C)ancel: ").strip().lower()
            if choice == 'c':
                print(" Command selection cancelled")
                return None

            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(commands):
                    return commands[choice_idx]
                else:
                    print(f" Invalid selection. Please choose a number between 1 and {len(commands)}")
            except ValueError:
                print(" Please enter a number or 'c' to cancel")
    elif len(commands) == 1:
        return commands[0]
    else:
        print(" No valid commands found in Mistral's response")
        return None

def execute_command(command):
    print("\n Command Ready to Execute:")
    if len(command) > 60:
        # For long commands, show it on multiple lines
        print(f" {command[:60]}...")
        print(f" ...{command[60:]}")
    else:
        print(f" {command}")

    confirmation = input("\nExecute this command? (Y)es / (N)o / (E)dit: ").strip().lower()
    if confirmation == 'y':
        print("\n Executing command...\n")
        subprocess.run(command, shell=True)
        print("\n Command execution complete")
    elif confirmation == 'e':
        edited = input("\n Enter modified command: ").strip()
        if edited:
            print("\n Executing modified command...\n")
            subprocess.run(edited, shell=True)
            print("\n Command execution complete")
    else:
        print("\n Command cancelled")

def test_microphone(input_device_index=None):
    """
    Test if the microphone is working and capturing audio
    """
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

    max_volume = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
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

def main():
    # 1) Load plugins
    loaded_plugins = load_plugins()

    # 2) Ask the user which mic to use
    mic_index = select_microphone()

    # 3) Main menu
    while True:
        print("\n Choose an option:")
        print(" (T) Type command")
        print(" (V) Voice command")
        print(" (M) Test microphone")
        print(" (Q) Quit")

        choice = input("\nEnter choice (T/V/M/Q): ").strip().lower()

        if choice == "t":
            command = input("Enter command: ").strip().lower()
            # Pass through plugins first
            plugin_output = process_command_with_plugins(command, loaded_plugins)
            # Then process with Mistral (or your existing command processing)
            corrected = process_with_mistral(plugin_output)
            if corrected:
                execute_command(corrected)
        elif choice == "v":
            command = get_voice_command(input_device_index=mic_index)
            if command:
                # Pass through plugins first
                plugin_output = process_command_with_plugins(command, loaded_plugins)
                # Then process with Mistral (or your existing command processing)
                corrected = process_with_mistral(plugin_output)
                if corrected:
                    execute_command(corrected)
        elif choice == "m":
            test_microphone(input_device_index=mic_index)
        elif choice == "q":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Loading Whisper model (one-time operation)...")
    model = whisper.load_model("tiny")
    print("Model loaded successfully!")
    main() 