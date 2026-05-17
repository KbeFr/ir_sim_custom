#!/usr/bin/env python3
"""
Generate random obstacle configurations for robotic simulation.
Usage: python gen_obstacles.py [options]
"""
import argparse
import math
import random

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Generate random obstacles for robot sim")
    p.add_argument("--width",      type=float, default=40,    help="World width")
    p.add_argument("--height",     type=float, default=40,    help="World height")
    p.add_argument("--scarcity",   type=float, default=0.5,   help="Scarcity 0=dense, 1=sparse")
    p.add_argument("--robot-r",    type=float, default=0.3,   help="Robot radius (clearance)")
    p.add_argument("--seed",       type=int,   default=None,  help="Random seed")
    p.add_argument("--corners",    action="store_true",        help="Add corner circle markers")
    p.add_argument("--output",     type=str,   default="obstacles.yaml")
    return p.parse_args()

def rect_corners(cx, cy, l, w, angle):
    """Return the 4 corners of a rotated rectangle."""
    hw, hl = w / 2, l / 2
    pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + ca*x - sa*y, cy + sa*x + ca*y) for x, y in pts]

def aabb(corners):
    xs, ys = zip(*corners)
    return min(xs), max(xs), min(ys), max(ys)

def circles_overlap(cx1, cy1, r1, cx2, cy2, r2, margin=0.0):
    return math.hypot(cx2-cx1, cy2-cy1) < r1 + r2 + margin

def rect_circle_overlap(corners, cx, cy, r, margin=0.0):
    xmin, xmax, ymin, ymax = aabb(corners)
    # Approximate: expand AABB by radius and check point
    return not (cx + r + margin < xmin or cx - r - margin > xmax or
                cy + r + margin < ymin or cy - r - margin > ymax)

def place_obstacle(obs_type, placed, W, H, margin):
    """Try to place one obstacle without overlap. Returns obstacle dict or None."""
    for _ in range(200):
        if obs_type == "circle":
            r   = random.uniform(0.4, 1.5)
            cx  = random.uniform(r + margin, W - r - margin)
            cy  = random.uniform(r + margin, H - r - margin)
            # Check overlap
            ok = True
            for p in placed:
                if p["kind"] == "circle":
                    if circles_overlap(cx, cy, r, p["cx"], p["cy"], p["r"], margin):
                        ok = False; break
                else:
                    if rect_circle_overlap(p["corners"], cx, cy, r, margin):
                        ok = False; break
            if ok:
                return {"kind": "circle", "cx": cx, "cy": cy, "r": r,
                        "yaml": {"shape": {"name": "circle", "radius": round(r,2),
                                           "center": [0,0]},
                                 "state": [round(cx,2), round(cy,2), 0]}}
        else:  # rectangle
            l     = random.uniform(0.8, 4.0)
            w     = random.uniform(0.8, 3.0)
            angle = random.uniform(0, math.pi)
            # Keep away from borders using half-diagonal as safe margin
            hd    = math.hypot(l, w) / 2
            cx    = random.uniform(hd + margin, W - hd - margin)
            cy    = random.uniform(hd + margin, H - hd - margin)
            corners = rect_corners(cx, cy, l, w, angle)
            ok = True
            for p in placed:
                if p["kind"] == "circle":
                    if rect_circle_overlap(corners, p["cx"], p["cy"], p["r"], margin):
                        ok = False; break
                else:
                    # AABB vs AABB approximation
                    xmin1,xmax1,ymin1,ymax1 = aabb(corners)
                    xmin2,xmax2,ymin2,ymax2 = aabb(p["corners"])
                    if not (xmax1+margin < xmin2 or xmin1-margin > xmax2 or
                            ymax1+margin < ymin2 or ymin1-margin > ymax2):
                        ok = False; break
            if ok:
                return {"kind": "rect", "corners": corners,
                        "yaml": {"shape": {"name": "rectangle",
                                           "length": round(l,2), "width": round(w,2)},
                                 "state": [round(cx,2), round(cy,2), round(angle,2)]}}
    return None

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    W, H = args.width, args.height
    margin = args.robot_r * 2  # minimum gap = robot diameter

    # Number of obstacles: scarcity 0 → dense, 1 → sparse
    area   = W * H
    max_n  = int(area / 20)   # ~1 obstacle per 20 sq units at densest
    min_n  = int(area / 200)  # ~1 per 200 at sparsest
    n      = max(1, int(min_n + (1 - args.scarcity) * (max_n - min_n)))

    placed, yaml_obs = [], []
    types = ["circle", "rectangle"]

    for _ in range(n):
        kind = random.choice(types)
        obs  = place_obstacle(kind, placed, W, H, margin)
        if obs:
            placed.append(obs)
            yaml_obs.append(obs["yaml"])

    if args.corners:
        for cx, cy in [(1,1),(W-1,1),(1,H-1),(W-1,H-1)]:
            yaml_obs.append({"shape": {"name":"circle","radius":1.0,"center":[0,0]},
                             "state": [cx, cy, 0]})

    out = {"obstacle": yaml_obs}
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Generated {len(yaml_obs)} obstacles → {args.output}")

if __name__ == "__main__":
    main()
