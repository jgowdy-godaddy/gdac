#!/usr/bin/env python3
"""
GoDaddy ASCII art with gradient colors and image support.
"""

import os
import sys
import base64
import subprocess
import termios
import tty
import select
from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.style import Style
try:
    from .terminal_detect import detect_terminal_image_support
except ImportError:
    from terminal_detect import detect_terminal_image_support

# GoDaddy brand colors (teal gradient)
COLORS = [
    "#00CBCD",  # Light teal
    "#00B4B6",  
    "#009D9F",
    "#008688",
    "#006F71",
    "#00585A",
    "#004143",
]

def detect_terminal_background() -> str:
    """Detect if terminal has dark or light background."""
    # Check COLORFGBG first (fastest)
    if os.environ.get('COLORFGBG'):
        try:
            parts = os.environ['COLORFGBG'].split(';')
            if len(parts) >= 2:
                bg = parts[-1]
                if bg.isdigit():
                    bg_num = int(bg)
                    if bg_num == 7 or bg_num >= 9:
                        return 'light'
                    else:
                        return 'dark'
        except:
            pass
    
    # Try to query the terminal's actual foreground color
    try:
        if sys.stdin.isatty() and sys.stdout.isatty():
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Set terminal to raw mode
                tty.setraw(sys.stdin.fileno())
                
                # Query foreground color (OSC 10) 
                sys.stdout.write('\033]10;?\007')
                sys.stdout.flush()
                
                # Wait for response (with timeout)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    response = ''
                    while True:
                        char = sys.stdin.read(1)
                        response += char
                        if char == '\\' or len(response) > 100:
                            break
                    
                    # Parse the RGB response
                    # Format: ESC]10;rgb:RRRR/GGGG/BBBB\ESC\\
                    if 'rgb:' in response:
                        rgb_part = response.split('rgb:')[1].split('\\')[0]
                        parts = rgb_part.split('/')
                        if len(parts) == 3:
                            # Convert to 0-255 range (response is in hex)
                            r = int(parts[0][:2], 16) if len(parts[0]) >= 2 else 0
                            g = int(parts[1][:2], 16) if len(parts[1]) >= 2 else 0
                            b = int(parts[2][:2], 16) if len(parts[2]) >= 2 else 0
                            
                            # Calculate brightness of foreground
                            # If foreground is light, background is probably dark
                            brightness = (r * 299 + g * 587 + b * 114) / 1000
                            if brightness > 128:
                                return 'dark'  # Light text = dark background
                            else:
                                return 'light'  # Dark text = light background
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except:
        pass
    
    # Default to dark
    return 'dark'

def display_godaddy_logo_image(console: Console = None) -> bool:
    """Try to display GoDaddy logo image if terminal supports it."""
    supports, protocol = detect_terminal_image_support()
    
    if not supports:
        return False
    
    # Choose logo based on background color
    bg_type = detect_terminal_background()
    if bg_type == 'light':
        logo_path = Path(__file__).parent / "assets" / "godaddy_logo_light_bg.png"
    else:
        logo_path = Path(__file__).parent / "assets" / "godaddy_logo_dark_bg.png"
    
    # Fallback to generic logo if specific one doesn't exist
    if not logo_path.exists():
        logo_path = Path(__file__).parent / "assets" / "godaddy_logo.png"
    
    if not logo_path.exists():
        return False
    
    try:
        if protocol == "kitty":
            # Kitty graphics protocol
            subprocess.run(['kitty', 'icat', '--align', 'center', str(logo_path)], 
                         check=True, capture_output=True)
            return True
            
        elif protocol == "iterm2":
            # iTerm2 inline images
            with open(logo_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # iTerm2 escape sequence
            osc = '\033]'
            st = '\007'
            # Add newline before and after for spacing
            if console:
                console.print()
            print(f'{osc}1337;File=inline=1;preserveAspectRatio=1:{img_data}{st}')
            if console:
                console.print()
            return True
            
        elif protocol == "sixel":
            # Sixel graphics with transparent PNG
            result = subprocess.run(['img2sixel', '-w', '400', str(logo_path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                return True
    except Exception:
        pass
    
    return False

def print_godaddy_banner(console: Console = None):
    """Print the GoDaddy banner - image if supported, ASCII fallback."""
    if console is None:
        console = Console()
    
    # Try to display image first
    if display_godaddy_logo_image(console):
        return
    
    # Fall back to ASCII art
    
    # Simpler, cleaner ASCII art - properly aligned
    lines = [
        "",
        "  ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██████╗ ██╗   ██╗",
        " ██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝",
        " ██║  ███╗██║   ██║██║  ██║███████║██║  ██║██║  ██║ ╚████╔╝ ",
        " ██║   ██║██║   ██║██║  ██║██╔══██║██║  ██║██║  ██║  ╚██╔╝  ",
        " ╚██████╔╝╚██████╔╝██████╔╝██║  ██║██████╔╝██████╔╝   ██║   ",
        "  ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝   ",
        "",
    ]
    
    # Use single color and bypass Rich's formatting to avoid alignment issues
    banner_color = COLORS[2]  # Mid-range teal
    
    for line in lines:
        if not line.strip():
            print()  # Use plain print for empty lines
        else:
            # Use ANSI codes directly to avoid Rich's width calculations
            print(f"\033[1;36m{line}\033[0m")  # Cyan color with bold

# Legacy functions for compatibility
def get_godaddy_ascii_art():
    return []

def get_godaddy_with_logo():
    return []

def print_simple_godaddy(console: Console = None):
    print_godaddy_banner(console)

if __name__ == "__main__":
    console = Console()
    print_godaddy_banner(console)