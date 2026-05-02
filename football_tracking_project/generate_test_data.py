import pandas as pd
import numpy as np

def generate_data(num_frames=200):
    records = []
    objects = [
        [0,  1,  100, 200,  2.0,  0.5],
        [0,  2,  300, 180,  1.5,  0.2],
        [1,  3,  800, 190,  1.0,  0.3],
        [1,  4,  950, 210,  0.8, -0.1],
        [2,  5,  530, 300,  0.5,  0.1],
        [32, 99, 520, 380,  5.0,  2.0],
    ]
    rng = np.random.default_rng(42)
    for frame in range(1, num_frames + 1):
        for obj in objects:
            team_id, player_id, x, y, vx, vy = obj
            nx = x + vx * (frame - 1) + rng.normal(0, 0.5)
            ny = y + vy * (frame - 1) + rng.normal(0, 0.5)
            records.append({
                "frame":      frame,
                "team_id":    team_id,
                "player_id":  player_id,
                "bb_left":    round(nx, 2),
                "bb_top":     round(ny, 2),
                "bb_width":   60  if team_id != 32 else 22,
                "bb_height":  120 if team_id != 32 else 22,
                "confidence": round(0.9 + rng.uniform(0, 0.09), 4),
                "extra":      -1,
            })
    df = pd.DataFrame(records)
    df.to_csv("data/dummy_detections_200f.csv", index=False)
    print(f"[OK] Generated {num_frames} frames -> data/dummy_detections_200f.csv")

if __name__ == "__main__":
    generate_data(200)
