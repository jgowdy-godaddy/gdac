#!/usr/bin/env python3
"""
Detect terminal image display capabilities.
"""

import os
import sys
import subprocess
from typing import Optional, Tuple

def detect_terminal_image_support() -> Tuple[bool, Optional[str]]:
    """
    Detect if the terminal supports displaying images.
    
    Returns:
        (supports_images, protocol_name)
        
    Protocols detected:
        - "kitty": Kitty graphics protocol
        - "iterm2": iTerm2 inline images protocol  
        - "sixel": Sixel graphics (various terminals)
        - "terminology": Terminology terminal
        - None: No image support detected
    """
    
    # Check environment variables
    term = os.environ.get('TERM', '').lower()
    term_program = os.environ.get('TERM_PROGRAM', '').lower()
    
    # 1. Check for VS Code terminal (no image support)
    if term_program == 'vscode':
        return False, None
    
    # 2. macOS Terminal.app (no image support)
    if 'apple' in term_program and 'terminal' in term_program:
        return False, None
    
    # 3. Check for Kitty
    if 'kitty' in term or os.environ.get('KITTY_WINDOW_ID'):
        return True, "kitty"
    
    # 4. Check for iTerm2
    if 'iterm' in term_program or os.environ.get('ITERM_SESSION_ID'):
        return True, "iterm2"
    
    # 5. Check for WezTerm (supports both iTerm2 and Sixel)
    if 'wezterm' in term or os.environ.get('WEZTERM_EXECUTABLE'):
        return True, "iterm2"  # WezTerm prefers iTerm2 protocol
    
    # 6. Check for Terminology
    if 'terminology' in term:
        return True, "terminology"
    
    # 7. Check for Windows Terminal (supports Sixel in newer versions)
    if os.environ.get('WT_SESSION'):
        # Windows Terminal 1.22+ supports Sixel
        return True, "sixel"
    
    # 8. Skip Sixel detection for now - it can hang terminals
    # if _check_sixel_support():
    #     return True, "sixel"
    
    # Default: no image support
    return False, None

# Removed _check_sixel_support as it can leave terminal in bad state

def get_terminal_info() -> dict:
    """Get detailed terminal information."""
    info = {
        'TERM': os.environ.get('TERM', 'unknown'),
        'TERM_PROGRAM': os.environ.get('TERM_PROGRAM', 'unknown'),
        'supports_images': False,
        'image_protocol': None,
        'terminal_name': 'unknown'
    }
    
    supports, protocol = detect_terminal_image_support()
    info['supports_images'] = supports
    info['image_protocol'] = protocol
    
    # Determine terminal name
    if os.environ.get('KITTY_WINDOW_ID'):
        info['terminal_name'] = 'Kitty'
    elif os.environ.get('ITERM_SESSION_ID'):
        info['terminal_name'] = 'iTerm2'
    elif os.environ.get('WEZTERM_EXECUTABLE'):
        info['terminal_name'] = 'WezTerm'
    elif os.environ.get('WT_SESSION'):
        info['terminal_name'] = 'Windows Terminal'
    elif info['TERM_PROGRAM'] == 'vscode':
        info['terminal_name'] = 'VS Code'
    elif info['TERM_PROGRAM'] == 'Apple_Terminal':
        info['terminal_name'] = 'macOS Terminal'
    elif 'terminology' in info['TERM']:
        info['terminal_name'] = 'Terminology'
    
    return info

def display_image_if_supported(image_path: str) -> bool:
    """
    Display an image if the terminal supports it.
    
    Returns True if image was displayed, False otherwise.
    """
    supports, protocol = detect_terminal_image_support()
    
    if not supports:
        return False
    
    if protocol == "kitty":
        # Use Kitty's icat command if available
        try:
            subprocess.run(['kitty', 'icat', image_path], check=True)
            return True
        except:
            pass
    
    elif protocol == "iterm2":
        # Use iTerm2's inline images protocol
        import base64
        try:
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # iTerm2 proprietary escape sequence
            osc = '\033]'
            st = '\007'
            print(f'{osc}1337;File=inline=1:{img_data}{st}')
            return True
        except:
            pass
    
    elif protocol == "sixel":
        # Try to use img2sixel if available
        try:
            subprocess.run(['img2sixel', image_path], check=True)
            return True
        except:
            pass
    
    return False

if __name__ == "__main__":
    # Test terminal detection
    info = get_terminal_info()
    
    print("Terminal Information:")
    print(f"  Terminal: {info['terminal_name']}")
    print(f"  TERM: {info['TERM']}")
    print(f"  TERM_PROGRAM: {info['TERM_PROGRAM']}")
    print(f"  Supports Images: {info['supports_images']}")
    
    if info['supports_images']:
        print(f"  Image Protocol: {info['image_protocol']}")
        print("\n  Your terminal can display images!")
        print(f"  Protocol to use: {info['image_protocol']}")
    else:
        print("\n  Your terminal does not support inline images.")
        print("  Terminals with image support:")
        print("    • Kitty (best support)")
        print("    • iTerm2 (macOS)")
        print("    • WezTerm (cross-platform)")
        print("    • Windows Terminal 1.22+ (Sixel)")
        print("    • Terminology (Linux)")