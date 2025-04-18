# Voice Command CLI (voice-cmd)

This project provides a command-line interface (CLI) that allows you to execute shell commands using either your voice or typed text. It leverages the OpenAI Whisper model for speech-to-text transcription and Mistral AI for command suggestions and refinement. It also features a simple plugin system for pre-processing commands.

## Features

*   **Voice Control:** Speak commands naturally and have them transcribed using Whisper.
*   **Text Input:** Fallback option to type commands directly.
*   **AI Command Suggestions:** Uses Mistral AI to interpret your intent and suggest appropriate shell commands with explanations.
*   **Command Execution:** Executes the selected command in your detected shell (PowerShell recommended on Windows).
*   **Microphone Selection:** Choose your preferred input microphone.
*   **Microphone Testing:** Utility to check if your selected microphone is capturing audio correctly.
*   **Intelligent Voice Recording:** Automatically starts when you speak and stops after detecting silence.
*   **Real-time Volume Visualization:** Visual feedback while recording with colored volume indicators.
*   **Plugin System:** Extend functionality by adding Python plugins to pre-process commands (e.g., spell checking).
*   **Dangerous Command Filtering:** Safety system to prevent execution of potentially harmful commands.
*   **Windows Focused:** Currently optimized and tested primarily for Windows environments (using PowerShell or CMD). Includes necessary patches (`patch_ctypes.py`) for compatibility.

## Requirements

*   **Python:** Version 3.8 or higher recommended.
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** For cloning the repository.
*   **ffmpeg:** Required by Whisper for audio processing.
    *   Download from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Extract the downloaded archive.
    *   **Add the `bin` directory** from the extracted folder to your system's **PATH environment variable**. You can verify the installation by opening a new terminal and typing `ffmpeg -version`.
*   **Mistral AI API Key:** You need an API key from [Mistral AI](https://mistral.ai/).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Create and Activate a Virtual Environment:**
    *   **Windows (cmd/powershell):**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:** (Note: Currently Windows-focused, may require adjustments)
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *(You should see `(venv)` at the beginning of your terminal prompt)*

3.  **Install Dependencies:**
    This command installs the required Python packages listed in `setup.py`, including Whisper, and also installs your project in "editable" mode.
    ```bash
    pip install -e .
    ```
    *(This might take some time, especially when downloading the Whisper model dependencies).*

4.  **Verify `ffmpeg`:** Ensure `ffmpeg` is accessible from your terminal by running `ffmpeg -version`. If not, revisit the PATH setup step.

## Running the Application

Make sure your virtual environment is activated (`(venv)` should be visible in your prompt).

Run the application using the command installed by `setup.py`:

## Using the Application

After launching the application, you will see a menu with the following options:

* **(T) Type command:** Enter a command via text input
* **(V) Voice command:** Speak a command via microphone
* **(M) Test microphone:** Test if your microphone is working properly
* **(D) Microphone diagnostic:** Run a detailed microphone diagnostic
* **(Q) Quit:** Exit the application

When using voice commands:
1. The system will perform a quick microphone check
2. Follow the countdown instructions
3. Speak clearly after the "GO! SPEAK NOW..." prompt
4. The recording will automatically stop after detecting silence
5. The system will transcribe your speech and offer command suggestions
6. Select the appropriate command option or cancel

The application includes a plugin system that can automatically correct common typos or words (e.g., "filez" → "files"). These plugins are loaded from the `plugins` directory.

## Safety Features

The application includes several safety features:
* Filtering of potentially dangerous commands
* Option to review commands before execution
* Ability to cancel command execution

## Troubleshooting

If you experience microphone issues:
* Ensure your microphone is not muted (check physical mute buttons)
* Verify Windows has granted microphone permissions to the application
* Try using the (M) option to test your microphone
* Run the (D) option for a detailed microphone diagnostic

If Whisper model fails to load:
* Ensure you have sufficient disk space and memory
* Check that ffmpeg is properly installed and accessible via PATH