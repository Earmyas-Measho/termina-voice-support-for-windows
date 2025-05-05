import os
import time
import whisper

def test_wav_file(wav_file_path, whisper_model=None):

    if not os.path.exists(wav_file_path):
        print(f"File {wav_file_path} does not exist.")
        return None
    
    if not wav_file_path.lower().endswith('.wav'):
        print(f"File {wav_file_path} is not a .wav file.")
        return None
    if whisper_model is None:
        try:
            from voice_clip import whisper_model
            if whisper_model is None:
                print("Loading Whisper model (small)...")
                whisper_model = whisper.load_model("small")
                print("Model loaded")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
        
    # Process the file
    print(f"Processing: {os.path.basename(wav_file_path)}...")

    try:
        # Time the transcription
        start_time = time.time()
        
        # Transcribe the audio
        result = whisper_model.transcribe(wav_file_path, language="en", fp16=False)
        
        # Get the text
        transcribed_text = result["text"].strip()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Display result
        print(f"Completed in {processing_time:.1f} seconds")
        print(f"\nTranscription result:")
        print(f"\"{transcribed_text}\"")
        
        return transcribed_text
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
    
def select_wav_file():
    print("\nSelect a WAV file to test:")
    print("1. Enter file path")
    print("2. Cancel")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        # Manual path entry
        file_path = input("Enter WAV file path: ").strip()
        file_path = file_path.strip('"\'')  # Remove quotes if any
        
        if not file_path:
            print("No path entered. Cancelled.")
            return None
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        if not file_path.lower().endswith('.wav'):
            print(f"Not a WAV file: {file_path}")
            return None
            
        return file_path
    
    elif choice == "2":
        # Cancel option
        print("Operation cancelled.")
        return None

def main():
    """Simple main function when running directly"""
    print("\nSimple WAV File Tester")
    print("=====================")
    
    # Get WAV file
    wav_file = select_wav_file()
    if not wav_file:
        print("No file selected. Exiting.")
        return
    
    # Test the file
    result = test_wav_file(wav_file)
    
    if result:
        # Ask if user wants to use this as a command
        process = input("\nProcess this as a command? (y/n): ").strip().lower()
        if process == 'y':
            # Import from voice_clip only when needed
            try:
                from voice_clip import process_command_with_plugins, process_with_mistral, execute_command, detect_shell, loaded_plugins
                
                # Get the shell
                shell = detect_shell()
                
                # Process the command
                plugin_output = process_command_with_plugins(result, loaded_plugins)
                command_options = process_with_mistral(plugin_output, shell)
                
                # Handle the results
                if command_options:
                    # Display options like in voice_clip.py
                    print("\nCommand Options:")
                    for i, option in enumerate(command_options, 1):
                        print(f"{i}. {option}")
                    
                    # Let user select
                    try:
                        choice = input("\nSelect option (1-n) or C to cancel: ").strip().lower()
                        if choice != 'c':
                            idx = int(choice) - 1
                            if 0 <= idx < len(command_options):
                                execute_command(command_options[idx])
                    except ValueError:
                        print("Invalid selection.")
                else:
                    print("Could not process command.")
            except ImportError as e:
                print(f"Error importing from voice_clip: {e}")
                print("Make sure this script is in the same directory as voice_clip.py")

if __name__ == "__main__":
    main()
    