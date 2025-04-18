from setuptools import setup, find_packages

setup(
    name="voice-cmd",
    version="0.1.0",
    py_modules=["voice_clip"],
    packages=find_packages(),
    install_requires=[
        "whisper",
        "numpy",
        "pyaudio",
        "requests",
        "pyttsx3",
    ],
    entry_points={
        "console_scripts": [
            "voice=voice_clip:main",
        ],
    },
) 