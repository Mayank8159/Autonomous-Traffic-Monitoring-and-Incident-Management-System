# src/gui.py
# -----------------------------------------------------------------
# Pygame GUI visualiser has been replaced by a web-based visualiser.
# See web_visualizer/visualizer.html for the new web interface.
# The rollout data format is JSON exported from main.py
# (results/rl_rollout.json), which the web visualizer reads.
# -----------------------------------------------------------------


# If anyone imports from this module, provide a minimal stub so imports
# don't fail. The actual visualisation is now handled by the web UI.
def draw_intersection(*args, **kwargs):
    """
    Stub function — the Pygame visualiser has been removed.
    Use web_visualizer/visualizer.html instead.
    """
    import warnings
    warnings.warn(
        "draw_intersection is a stub; the pygame GUI has been replaced "
        "by a web-based visualiser at web_visualizer/visualizer.html",
        FutureWarning,
    )
    return None