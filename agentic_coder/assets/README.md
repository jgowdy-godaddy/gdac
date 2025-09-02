# GoDaddy Logo Assets

The GoDaddy logo is displayed in terminals that support image protocols (Kitty, iTerm2, WezTerm, etc.).

## Logo File
- `godaddy_logo.png` - Pre-converted PNG with transparency (399x100, 8-bit RGBA)
- Source: Converted from `GD_LOCKUP_RGB_BW_DARK_BG.svg`

## Display Size
- Rendered at approximately 400x100 pixels
- Colors: GoDaddy teal gradient (#00CBCD to #004143)

## Supported Terminals
- **Kitty**: Full support via kitty graphics protocol
- **iTerm2**: Inline images on macOS
- **WezTerm**: Supports iTerm2 protocol
- **Windows Terminal 1.22+**: Sixel graphics
- **Terminology**: Native image support

## Fallback
If the terminal doesn't support image protocols, ASCII art with gradient colors is displayed instead.