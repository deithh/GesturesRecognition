"""
WandSense IMU - Master Design System & UI Theme for Pygame & Terminal
Generated according to ui-ux-pro-max: HUD / Data-Dense Dark Dashboard
"""
import pygame
import math

# =============================================================================
# COLOR TOKENS (WCAG 7:1+ contrast on dark obsidian background)
# =============================================================================
BG_VOID           = (10, 14, 23)        # #0A0E17 Deep obsidian background
BG_PANEL          = (16, 22, 36)        # #101624 Viewport & side panels
SURFACE_CARD      = (20, 28, 45)        # #141C2D Elevated bento card
SURFACE_CARD_ALT  = (24, 34, 54)        # #182236 Highlighted / active card
SURFACE_TRACK     = (28, 38, 60)        # #1C263C Track background for progress bars
SURFACE_PILL      = (24, 33, 52)        # #182134 Pill badge container

BORDER_BASE       = (38, 52, 78)        # #26344E Subtle card border
BORDER_FOCUS      = (0, 216, 246)       # #00D8F6 Cyber Cyan focus ring
BORDER_MUTED      = (28, 38, 58)        # #1C263A Internal divider lines

TEXT_PRIMARY      = (243, 246, 251)     # #F3F6FB 95% White
TEXT_SECONDARY    = (154, 169, 192)     # #9AA9C0 Muted technical slate
TEXT_MUTED        = (96, 112, 136)      # #607088 Tertiary details / hints

ACCENT_CYAN       = (0, 216, 246)       # #00D8F6 Primary telemetry brand
ACCENT_EMERALD    = (16, 185, 129)      # #10B981 Success / Confirmed / Calibrated
ACCENT_AMBER      = (245, 158, 11)      # #F59E0B Motion in progress / Warning
ACCENT_CORAL      = (239, 68, 68)       # #EF4444 Kinetic threshold / Alert
ACCENT_VIOLET     = (139, 92, 246)      # #8B5CF6 Deep learning model engine
ACCENT_GOLD       = (251, 191, 36)      # #FBBF24 Motion trail apex

# 3D Elements Colors
WAND_GRIP         = (65, 80, 105)       # Matte gunmetal handle
WAND_SHAFT        = (215, 125, 35)      # Carbon bronze body
WAND_TIP_AURA     = (255, 140, 50)      # Neon tip aura
WAND_TIP_CORE     = (255, 250, 240)     # Plasma white core
GRID_LINE         = (28, 38, 56)        # Horizon grid
GRID_AXIS_X       = (244, 63, 94)       # Rose/Red X-axis
GRID_AXIS_Y       = (34, 197, 94)       # Emerald Green Y-axis
GRID_AXIS_Z       = (59, 130, 246)      # Electric Blue Z-axis


# =============================================================================
# TYPOGRAPHY SYSTEM
# =============================================================================
class Typography:
    def __init__(self):
        # Monospace & Sans fonts available on Windows / Linux / macOS
        mono_candidates = ["consolas", "fira code", "cascadia code", "dejavu sans mono", "courier new"]
        sans_candidates = ["segoe ui", "trebuchet ms", "arial", "helvetica", "dejavu sans"]

        self.font_title   = pygame.font.SysFont(sans_candidates, 22, bold=True)
        self.font_heading = pygame.font.SysFont(sans_candidates, 16, bold=True)
        self.font_body    = pygame.font.SysFont(sans_candidates, 14)
        self.font_badge   = pygame.font.SysFont(sans_candidates, 12, bold=True)
        self.font_mono    = pygame.font.SysFont(mono_candidates, 13, bold=True)
        self.font_mono_sm = pygame.font.SysFont(mono_candidates, 11)


# =============================================================================
# REUSABLE PYGAME RENDER HELPERS
# =============================================================================
def draw_card(surface, rect, bg_color=SURFACE_CARD, border_color=BORDER_BASE, radius=8, border_w=1):
    """Draws a modern rounded bento card with subtle border."""
    pygame.draw.rect(surface, bg_color, rect, border_radius=radius)
    if border_w > 0 and border_color is not None:
        pygame.draw.rect(surface, border_color, rect, width=border_w, border_radius=radius)


def draw_pill_badge(surface, text, font, x, y, bg_color=SURFACE_PILL, text_color=TEXT_PRIMARY,
                    border_color=BORDER_BASE, dot_color=None, padding_x=10, padding_y=4):
    """Draws a sleek floating status capsule/pill with optional indicator dot."""
    txt_surf = font.render(text, True, text_color)
    tw, th = txt_surf.get_size()

    dot_space = 14 if dot_color is not None else 0
    w = tw + padding_x * 2 + dot_space
    h = th + padding_y * 2
    rect = pygame.Rect(x, y, w, h)

    pygame.draw.rect(surface, bg_color, rect, border_radius=h // 2)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, width=1, border_radius=h // 2)

    content_x = x + padding_x
    if dot_color is not None:
        dot_center = (content_x + 4, y + h // 2)
        pygame.draw.circle(surface, dot_color, dot_center, 4)
        content_x += dot_space

    surface.blit(txt_surf, (content_x, y + padding_y))
    return rect


def draw_progress_bar(surface, x, y, w, h, ratio, fill_color, bg_color=SURFACE_TRACK,
                      border_color=BORDER_BASE, radius=4):
    """Draws a smooth progress gauge bar with rounded edges."""
    rect_bg = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, bg_color, rect_bg, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect_bg, width=1, border_radius=radius)

    fill_w = max(0, min(w, int(ratio * w)))
    if fill_w > 0:
        rect_fill = pygame.Rect(x, y, fill_w, h)
        pygame.draw.rect(surface, fill_color, rect_fill, border_radius=radius)


def draw_glow_sphere(surface, center, base_radius, color_rgb):
    """Draws a multi-layered glowing plasma sphere for wand tip."""
    cx, cy = int(center[0]), int(center[1])
    # Layer 1: Outer soft aura
    aura_surf = pygame.Surface((base_radius * 6, base_radius * 6), pygame.SRCALPHA)
    pygame.draw.circle(aura_surf, (*color_rgb, 35), (base_radius * 3, base_radius * 3), base_radius * 2.8)
    pygame.draw.circle(aura_surf, (*color_rgb, 75), (base_radius * 3, base_radius * 3), base_radius * 1.8)
    surface.blit(aura_surf, (cx - base_radius * 3, cy - base_radius * 3), special_flags=pygame.BLEND_ALPHA_SDL2)

    # Layer 2: Core solid tip
    pygame.draw.circle(surface, color_rgb, (cx, cy), base_radius)
    # Layer 3: Hot specular center
    pygame.draw.circle(surface, WAND_TIP_CORE, (cx, cy), max(2, base_radius // 2))
