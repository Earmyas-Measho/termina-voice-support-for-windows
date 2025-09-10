"""
Accessibility Plugin for VoiceCLI
Provides enhanced accessibility features for users with disabilities.
"""

import re
import os
from typing import Dict, List

class Plugin:
    """
    Accessibility plugin that enhances commands for users with visual or motor impairments.
    """
    
    def __init__(self):
        self.accessibility_commands = {
            # Screen reader friendly commands
            "read": "Get-Content",
            "show": "Write-Host",
            "display": "Write-Host",
            
            # Motor impairment friendly shortcuts
            "quick list": "Get-ChildItem -Name",
            "quick dir": "dir /b",
            "current": "Get-Location",
            "where": "Get-Location",
            
            # High contrast friendly
            "big list": "Get-ChildItem | Format-Table -AutoSize",
            "detailed": "Get-ChildItem | Format-List",
        }
        
        self.audio_feedback_commands = [
            "echo", "Write-Host", "Get-Content", "dir", "Get-ChildItem"
        ]
    
    def execute(self, command: str) -> str:
        """
        Process command for accessibility enhancements.
        """
        original_command = command.lower().strip()
        
        # Replace accessibility shortcuts
        for shortcut, replacement in self.accessibility_commands.items():
            if shortcut in original_command:
                command = command.replace(shortcut, replacement)
                print(f"[AccessibilityPlugin] Enhanced command for accessibility: '{shortcut}' → '{replacement}'")
        
        # Add audio feedback indicators for screen readers
        if any(audio_cmd in command for audio_cmd in self.audio_feedback_commands):
            # Add a comment for TTS to announce
            if "PowerShell" in command or "Get-" in command:
                command += " # Audio feedback enabled"
            print(f"[AccessibilityPlugin] Audio feedback enabled for screen readers")
        
        return command
    
    def get_accessibility_info(self) -> Dict[str, str]:
        """Return information about accessibility features."""
        return {
            "name": "Accessibility Plugin",
            "version": "1.0.0",
            "description": "Enhances VoiceCLI for users with disabilities",
            "features": [
                "Screen reader friendly commands",
                "Motor impairment shortcuts", 
                "High contrast output options",
                "Audio feedback integration"
            ]
        }
