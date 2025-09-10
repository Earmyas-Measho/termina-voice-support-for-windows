"""
DevOps Plugin for VoiceCLI
Provides DevOps-specific command shortcuts and enhancements.
"""

import re
import os
from typing import Dict, List

class Plugin:
    """
    DevOps plugin that provides shortcuts for common DevOps tasks.
    """
    
    def __init__(self):
        self.devops_shortcuts = {
            # Git shortcuts
            "git status": "git status --porcelain",
            "git log": "git log --oneline -10",
            "git branch": "git branch -a",
            "git remote": "git remote -v",
            
            # Docker shortcuts
            "docker ps": "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
            "docker images": "docker images --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}'",
            "docker logs": "docker logs --tail 50",
            
            # System monitoring
            "system info": "Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory",
            "disk usage": "Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name='Size(GB)';Expression={[math]::Round($_.Size/1GB,2)}}, @{Name='FreeSpace(GB)';Expression={[math]::Round($_.FreeSpace/1GB,2)}}",
            "process list": "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, CPU, WorkingSet",
            
            # Network tools
            "ping test": "Test-NetConnection -ComputerName 8.8.8.8 -Port 53",
            "port check": "Test-NetConnection -ComputerName localhost -Port",
            
            # File operations
            "find large files": "Get-ChildItem -Recurse | Where-Object {$_.Length -gt 100MB} | Sort-Object Length -Descending | Select-Object -First 10 Name, @{Name='Size(MB)';Expression={[math]::Round($_.Length/1MB,2)}}",
            "clean temp": "Get-ChildItem $env:TEMP -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue",
        }
        
        self.safety_checks = [
            "rm -rf", "del /s", "format", "fdisk", "diskpart"
        ]
    
    def execute(self, command: str) -> str:
        """
        Process command for DevOps enhancements.
        """
        original_command = command.lower().strip()
        
        # Check for dangerous operations and add warnings
        for danger in self.safety_checks:
            if danger in original_command:
                print(f"[DevOpsPlugin] ⚠️ WARNING: Potentially dangerous operation detected: {danger}")
                print(f"[DevOpsPlugin] Consider using safer alternatives or double-check your command")
        
        # Replace DevOps shortcuts
        for shortcut, replacement in self.devops_shortcuts.items():
            if shortcut in original_command:
                command = command.replace(shortcut, replacement)
                print(f"[DevOpsPlugin] Applied DevOps shortcut: '{shortcut}' → enhanced command")
                break
        
        # Add DevOps-specific enhancements
        if "git" in original_command:
            command = self._enhance_git_command(command)
        elif "docker" in original_command:
            command = self._enhance_docker_command(command)
        elif "system" in original_command or "monitor" in original_command:
            command = self._enhance_system_command(command)
        
        return command
    
    def _enhance_git_command(self, command: str) -> str:
        """Enhance Git commands with better formatting."""
        if "git log" in command and "--oneline" not in command:
            command = command.replace("git log", "git log --oneline -10")
        elif "git status" in command and "--porcelain" not in command:
            command = command.replace("git status", "git status --porcelain")
        return command
    
    def _enhance_docker_command(self, command: str) -> str:
        """Enhance Docker commands with better formatting."""
        if "docker ps" in command and "--format" not in command:
            command = command.replace("docker ps", "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'")
        elif "docker images" in command and "--format" not in command:
            command = command.replace("docker images", "docker images --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}'")
        return command
    
    def _enhance_system_command(self, command: str) -> str:
        """Enhance system monitoring commands."""
        if "process" in command.lower() and "Get-Process" not in command:
            command = "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, CPU, WorkingSet"
        elif "disk" in command.lower() and "Get-WmiObject" not in command:
            command = "Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name='Size(GB)';Expression={[math]::Round($_.Size/1GB,2)}}, @{Name='FreeSpace(GB)';Expression={[math]::Round($_.FreeSpace/1GB,2)}}"
        return command
    
    def get_devops_info(self) -> Dict[str, str]:
        """Return information about DevOps features."""
        return {
            "name": "DevOps Plugin",
            "version": "1.0.0",
            "description": "DevOps shortcuts and enhancements for VoiceCLI",
            "features": [
                "Git command shortcuts",
                "Docker command enhancements",
                "System monitoring tools",
                "Network diagnostic commands",
                "File operation shortcuts",
                "Safety checks for dangerous operations"
            ]
        }
