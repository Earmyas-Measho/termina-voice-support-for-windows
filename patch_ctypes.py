# patch_ctypes.py
# This module applies the monkey patch for ctypes.CDLL on Windows

import sys
import platform
import ctypes

print("--- patch_ctypes.py executing ---", file=sys.stderr) # DEBUG

# Apply patch only on Windows
if platform.system() == "Windows":
    # Store the original CDLL function
    original_CDLL = ctypes.CDLL
    print(f"Original CDLL in patcher: {original_CDLL}", file=sys.stderr) # DEBUG

    # Define arguments based on Python version for compatibility
    def patched_CDLL(name, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=False, use_last_error=False, winmode=None):
        name_to_load = name
        # Check if the problematic None is being passed (likely by whisper)
        if name is None:
            print("!!! PATCHING CDLL (from patch_ctypes.py): Intercepted None, using 'msvcrt' instead !!!", file=sys.stderr)
            name_to_load = 'msvcrt' # Substitute the correct library name

        print(f"--- Patched CDLL (from patch_ctypes.py) called with name: {name_to_load} ---", file=sys.stderr) # DEBUG

        # Call the original CDLL with the potentially corrected name
        # Handle winmode argument difference in Python 3.8+
        try:
            if sys.version_info >= (3, 8):
                 return original_CDLL(name_to_load, mode=mode, handle=handle, use_errno=use_errno, use_last_error=use_last_error, winmode=winmode)
            else:
                 # winmode argument doesn't exist in older versions
                 return original_CDLL(name_to_load, mode=mode, handle=handle, use_errno=use_errno, use_last_error=use_last_error)
        except Exception as e:
            print(f"Error calling original CDLL with {name_to_load}: {e}", file=sys.stderr)
            raise # Re-raise the exception

    # Replace the original CDLL function with our patched version
    ctypes.CDLL = patched_CDLL
    print(f"Patched CDLL assigned in patcher: {ctypes.CDLL}", file=sys.stderr) # DEBUG
else:
    print("--- Not Windows, patch not applied ---", file=sys.stderr) # DEBUG 