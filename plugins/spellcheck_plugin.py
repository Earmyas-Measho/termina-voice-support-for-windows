import re

class Plugin:
    """
    A simple plugin that corrects common typos or placeholders in the voice command.
    """
    def execute(self, command: str) -> str:
        # Define your custom corrections here
        corrections = {
            "pls": "please",
            "filez": "files",
            "lsit": "list",
            "dirr": "dir",
        }

        corrected = command
        for wrong, right in corrections.items():
            # Replace any occurrence of 'wrong' with 'right'
            corrected = re.sub(rf"\b{wrong}\b", right, corrected)

        # Print a message if a correction was made
        if corrected != command:
            print(f"[SpellCheckPlugin] Corrected '{command}' → '{corrected}'")

        return corrected 