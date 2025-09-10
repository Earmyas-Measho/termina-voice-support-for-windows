from setuptools import setup, find_packages

setup(
    name="voice-cmd",
    version="0.1.0",
    py_modules=["voice_cli", "voice_clip", "patch_ctypes"],
    packages=find_packages(),
    install_requires=[
        "openai-whisper",
        "numpy",
        "pyaudio",
        "requests",
        "pyttsx3",  # Re-enabled for TTS functionality
    ],
    entry_points={
        "console_scripts": [
            "voice=voice_cli:main",
        ],
    },
) 