# src/gui.py
import pygame
import numpy as np

# Initialise pygame once at module level (but functions can re‑initialise if needed)
pygame.init()

# Some colour constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
GRAY = (200, 200, 200)

FONT = pygame.font.SysFont("Arial", 18)


def draw_intersection(screen, env, probs=None, selected_action=None):
    """
    Draw a very simple 2‑lane intersection visualisation.

    Parameters
    ----------
    screen : pygame.Surface
        The surface to draw on.
    env : TrafficEnv
        The environment instance – used to read queue lengths and current phase.
    probs : np.ndarray or None, shape (2,)
        Softmax probabilities for the two actions (NS green, EW green).
    selected_action : int or None
        The action actually taken (0 = NS green, 1 = EW green).
    """
    screen.fill(WHITE)

    # --- Draw roads ---
    road_width = 100
    # North‑south road (vertical line)
    pygame.draw.line(screen, GRAY, (400, 100), (400, 500), road_width)
    # East‑west road (horizontal line)
    pygame.draw.line(screen, GRAY, (100, 300), (700, 300), road_width)

    # --- Draw queues as rectangles ---
    # We have 4 lanes: N‑in, N‑out, E‑in, E‑out (order as in env.queue)
    # Position them around the centre (300,300) roughly.
    lane_positions = {
        0: (300 - road_width // 2, 200),   # N‑in (north top)
        1: (300 + road_width // 2, 400),   # E‑in (east right)
        2: (300 - road_width // 2, 400),   # N‑out (south bottom) – we'll just mirror
        3: (300, 300 - road_width // 2),   # W‑in (west left) – placeholder
    }

    # Scale queue length to a visible bar height (max shown as 50 pixels)
    max_bar = 50
    for idx, (x, y) in lane_positions.items():
        queue_len = float(env.queue[idx]) if idx < len(env.queue) else 0
        bar_h = int((queue_len / max(env.max_queue, 1)) * max_bar)
        color = GREEN if bar_h > 0 else GRAY
        # draw bar
        pygame.draw.rect(screen, color, (x, y - bar_h, road_width // 2 if idx in (0,2) else road_width, bar_h))
        # label
        label = FONT.render(f"L{idx+1}: {queue_len:.1f}", True, BLACK)
        screen.blit(label, (x, y - bar_h - 20))

    # --- Current phase indicator ---
    phase = env.phase  # 0 -> NS green, 1 -> EW green
    if phase == 0:
        phase_color = GREEN
        phase_text = "NS Green"
    else:
        phase_color = RED
        phase_text = "EW Green"

    pygame.draw.circle(screen, phase_color, (650, 50), 20)
    txt = FONT.render(phase_text, True, BLACK)
    screen.blit(txt, (680, 45))

    # --- Probabilities (if provided) ---
    if probs is not None:
        # Show as small text at top‑left
        prob_text = f"P(NS): {probs[0]:.2f}  P(EW): {probs[1]:.2f}"
        surf = FONT.render(prob_text, True, BLACK)
        screen.blit(surf, (20, 20))

    # --- Selected action ---
    if selected_action is not None:
        act_label = f"Action: {'NS' if selected_action == 0 else 'EW'} green"
        act_surf = FONT.render(act_label, True, BLACK)
        screen.blit(act_surf, (20, 50))

    # Update display
    pygame.display.flip()


# -----------------------------------------------------------------
# If this module is run directly, provide a tiny demo loop
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Very simple demo: create an env, train a dummy policy, then visualise.
    # This section is only for quick testing; normal usage imports the function.
    from src.env import TrafficEnv
    from src.model import MLPModel

    env = TrafficEnv()
    policy = MLPModel()
    # random just to have something to show
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            probs = policy(torch.tensor(env._get_state(), dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(probs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        env = TrafficEnv()  # reset for next step just for demo; real code would keep state

        draw_intersection(pygame.display.set_mode((800, 600)), env, probs.squeeze().numpy(), action)
        clock.tick(2)

    pygame.quit()