import os
import time
import sys

def batch_test_wav_files(directory_path="./resource", whisper_model=None, plugins=None, shell=None):
    """
    Test all WAV files in a directory using the existing manual testing workflow.
    """
    # Import required functions from voice_clip.py
    try:
        from voice_clip import (
            process_command_with_plugins,
            process_with_mistral,
            execute_command,
            detect_shell,
            load_plugins,
            load_whisper_model,
            load_config
        )
        
        # Load resources if not provided
        config = load_config()
        
        if whisper_model is None:
            print("Loading Whisper model...")
            whisper_model = load_whisper_model(config)
        
        if whisper_model is None:
            print("Error: Whisper model could not be loaded")
            return
            
        if plugins is None:
            print("Loading plugins...")
            plugins = load_plugins()
            
        if plugins is None or not plugins:
            print("Warning: No plugins loaded")
            plugins = {}
            
        if shell is None:
            print("Detecting shell...")
            shell = detect_shell()
        
    except ImportError as e:
        print(f"Error importing from voice_clip: {e}")
        print("Make sure this script is in the same directory as voice_clip.py")
        return
    
    # Normalize directory path
    directory_path = os.path.abspath(os.path.expanduser(directory_path))
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return
    
    # Find all WAV files
    wav_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.wav'):
            wav_files.append(os.path.join(directory_path, file))
    
    wav_files.sort()  # Sort alphabetically
    
    if not wav_files:
        print(f"No WAV files found in {directory_path}")
        return
    
    print(f"Found {len(wav_files)} WAV files in {directory_path}")
    
    # Process each file
    for i, wav_file in enumerate(wav_files, 1):
        file_name = os.path.basename(wav_file)
        print(f"\n[{i}/{len(wav_files)}] Testing: {file_name}")
        
        # Process the file using the same workflow as manual testing
        try:
            # Transcribe using Whisper
            start_time = time.time()
            result = whisper_model.transcribe(wav_file, language="en", fp16=False)
            transcription = result["text"].strip()
            
            # Show processing time
            processing_time = time.time() - start_time
            print(f"Transcription complete in {processing_time:.2f} seconds")
            print(f"\nTranscription: \"{transcription}\"")
            
            # Ask to process as command (just like manual flow)
            process = input("\nProcess this as a command? (y/n/s): ").strip().lower()
            
            if process == 's':  # Skip all remaining files
                print("Skipping remaining files...")
                break
                
            if process != 'y':
                continue
            
            # Process with plugins and Mistral
            plugin_output = process_command_with_plugins(transcription, plugins)
            command_options = process_with_mistral(plugin_output, shell)
            
            if command_options and isinstance(command_options, dict) and command_options.get("is_chain"):
                # Handle command chain
                print("\nCommand Chain detected:")
                for j, step in enumerate(command_options.get("steps", []), 1):
                    cmd = step.get("command", "N/A")
                    print(f"{j}. {cmd}")
                
                execute = input("\nExecute this command chain? (y/n): ").strip().lower()
                if execute == 'y':
                    from voice_clip import execute_command_chain
                    execute_command_chain(command_options)
            
            elif command_options:
                # Show command options
                print("\nCommand Options:")
                for j, option in enumerate(command_options, 1):
                    cmd = option.get('command', 'N/A')
                    explanation = option.get('explanation', '')
                    print(f"{j}. {cmd}")
                    print(f"   {explanation}")
                
                # Let user select option
                try:
                    choice = input(f"\nSelect option (1-{len(command_options)}) or (C)ancel: ").strip().lower()
                    if choice == 'c':
                        continue
                        
                    idx = int(choice) - 1
                    if 0 <= idx < len(command_options):
                        selected_command = command_options[idx].get('command')
                        if selected_command:
                            execute_command(selected_command)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input, continuing to next file")
            else:
                print("No command options generated")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nBatch testing complete")

def main():
    """Main function when run as standalone script"""
    print("\nWAV Batch Testing Utility")
    print("========================")
    
    # Get directory path
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input("Enter directory path [./resource]: ").strip()
        if not directory_path:
            directory_path = "./resource"
    
    # Run batch test without trying to access external variables
    batch_test_wav_files(directory_path)

if __name__ == "__main__":
    main()