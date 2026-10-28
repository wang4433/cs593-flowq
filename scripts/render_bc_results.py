"""Render saved BC evaluation episodes in the original simulator.

Replays recorded actions and verifies every observation, reward, and termination
against the recorded closed-loop evaluation. No training or new evaluation score.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flowq.data import sha256
from flowq.env import Task
from flowq.flow import fingerprint
from flowq.runner import load_flow, seed, write_json


FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    path = FONT_ROOT / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def frame_for(env, record, step, fps, end=False):
    views = [env.env.sim.render(width=640, height=480, camera_name=c)[::-1].copy()
             for c in ('sideview', 'agentview')]
    canvas = Image.new('RGB', (1280, 576), '#111827')
    canvas.paste(Image.fromarray(views[0]), (0, 64))
    canvas.paste(Image.fromarray(views[1]), (640, 64))
    draw = ImageDraw.Draw(canvas)
    outcome = 'SUCCESS' if record['success'] else 'TIMEOUT'
    color = '#66e0ae' if record['success'] else '#ffcb78'
    draw.text((18, 8), f'Lift | Frozen Flow BC | seed {record["seed"]}', font=font(24, True), fill='white')
    draw.text((1000, 8), f'Result: {outcome}', font=font(22, True), fill=color)
    draw.text((18, 39), 'Side view', font=font(16), fill='#cbd5e1')
    draw.text((658, 39), 'Gripper / object close-up', font=font(16), fill='#cbd5e1')
    status = 'END - final frame held' if end else f'step {step:03d}/{record["steps"]:03d} | simulation {step / 20:.2f}s'
    draw.text((18, 551), status, font=font(16), fill='white')
    draw.text((650, 551), f'Verified recorded-action replay | {fps / 20:.1f}x playback', font=font(16), fill='#cbd5e1')
    return np.asarray(canvas), views[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lift_bc_stage1.json')
    parser.add_argument('--flow', default='runs/lift_bc_stage1/bc/flow.pt')
    parser.add_argument('--evaluation', default='runs/lift_bc_stage1/eval')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 2, 7, 1])
    parser.add_argument('--fps', type=int, default=10, help='20 = real time; 10 = half speed')
    parser.add_argument('--out', default='runs/lift_bc_stage1/rendered')
    args = parser.parse_args()
    if args.fps < 1: parser.error('--fps must be positive')
    cfg = json.loads(Path(args.config).read_text())
    evaluation = Path(args.evaluation)
    results = json.loads((evaluation / 'summary.json').read_text())
    if sha256(args.flow) != results['checkpoint_sha256'] or cfg != results['config']:
        raise ValueError('Evaluation/config/checkpoint identities must match')
    if any(i < 0 or i >= len(results['episodes']) for i in args.episodes):
        raise ValueError('Invalid episode index')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    seed(cfg['seed']); flow, ck = load_flow(args.flow, cfg)
    before = fingerprint(flow); start = time.perf_counter()
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    env = Task(cfg, ck['manifest']['env'])
    manifest = []; final_frames = []; storyboard_rows = []
    writer_args = dict(fps=args.fps, codec='libx264', quality=8, pixelformat='yuv420p',
                       ffmpeg_params=['-movflags', '+faststart'])
    try:
        with imageio.get_writer(out / 'lift_results_montage.mp4', **writer_args) as montage:
            for idx in args.episodes:
                record = results['episodes'][idx]
                with np.load(evaluation / f'episode_{idx:03d}.npz') as saved:
                    states = saved['states']; actions = saved['actions']; rewards = saved['raw_rewards']
                s = env.reset(record['seed'])
                max_error = float(np.max(np.abs(s - states[0])))
                if max_error > 1e-7: raise AssertionError('Initial observation mismatch')
                context = MjRenderContextOffscreen(env.env.sim, device_id=-1)
                context.vopt.geomgroup[0] = 0; context.vopt.geomgroup[1] = 1
                result_name = 'success' if record['success'] else 'timeout'
                filename = f'lift_seed_{record["seed"]}_{result_name}.mp4'
                snapshots = {}; preview = []
                try:
                    with imageio.get_writer(out / filename, **writer_args) as video:
                        frame, closeup = frame_for(env, record, 0, args.fps)
                        video.append_data(frame); montage.append_data(frame); snapshots[0] = closeup
                        if idx == args.episodes[0]: preview.append(Image.fromarray(frame).resize((640, 288)))
                        for step, action in enumerate(actions, 1):
                            s, _, done, info = env.step(action)
                            max_error = max(max_error, float(np.max(np.abs(s - states[step]))))
                            if max_error > 1e-7 or abs(info['raw_reward'] - rewards[step-1]) > 1e-10:
                                raise AssertionError(f'Recorded trajectory mismatch at step {step}: {max_error}')
                            if done != (step == len(actions)):
                                raise AssertionError('Termination differs from saved episode')
                            frame, closeup = frame_for(env, record, step, args.fps)
                            video.append_data(frame); montage.append_data(frame)
                            if step in (len(actions)//2, len(actions)): snapshots[step] = closeup
                            if idx == args.episodes[0]: preview.append(Image.fromarray(frame).resize((640, 288)))
                        assert bool(info['success']) == record['success']
                        final, _ = frame_for(env, record, len(actions), args.fps, end=True)
                        for _ in range(args.fps):
                            video.append_data(final); montage.append_data(final)
                        final_frames.append(Image.fromarray(final))
                        if preview:
                            preview.extend([Image.fromarray(final).resize((640, 288))] * args.fps)
                            preview[0].save(out / 'lift_success_preview.gif', save_all=True,
                                append_images=preview[1:], duration=1000/args.fps, loop=0)
                    storyboard_rows.append((record, snapshots))
                    manifest.append(dict(episode_index=idx, seed=record['seed'],success=record['success'],
                        steps=len(actions),video=filename,max_abs_observation_error=max_error,
                        recorded_states_actions_rewards_and_termination_matched=True,
                        video_frames=len(actions)+1+args.fps, fps=args.fps))
                    print(json.dumps(manifest[-1]), flush=True)
                    write_json(out / 'rendered_episodes.json', manifest)
                finally:
                    env.env.sim._render_context_offscreen = None
                    del context
                    gc.collect()
    finally:
        env.close()
    assert fingerprint(flow) == before == results['flow_fingerprint']
    grid = Image.new('RGB', (1280, 288*((len(final_frames)+1)//2)), '#111827')
    for i, frame in enumerate(final_frames): grid.paste(frame.resize((640, 288)), ((i%2)*640, (i//2)*288))
    grid.save(out / 'results_overview.png')
    storyboard = Image.new('RGB', (960, len(storyboard_rows)*276), '#111827')
    for row, (record, snapshots) in enumerate(storyboard_rows):
        draw = ImageDraw.Draw(storyboard)
        outcome = 'SUCCESS' if record['success'] else 'TIMEOUT'
        draw.text((12, row*276+6), f'Seed {record["seed"]} | {outcome} | Start / Midpoint / End', font=font(18, True), fill='white')
        for col, (_, closeup) in enumerate(sorted(snapshots.items())):
            storyboard.paste(Image.fromarray(closeup).resize((320, 240)), (col*320, row*276+36))
    storyboard.save(out / 'trajectory_storyboard.png')
    write_json(out / 'render_manifest.json', dict(source_evaluation=str(evaluation),
        source_checkpoint_sha256=results['checkpoint_sha256'],source_summary_sha256=sha256(evaluation/'summary.json'),
        render_script_sha256=sha256(__file__),mode='recorded action replay; all states checked',
        simulator_steps_replayed=sum(x['steps'] for x in manifest),new_evaluation_episodes=0,
        frozen_unchanged=True,playback_speed=args.fps/20,episodes=manifest,
        elapsed_seconds=time.perf_counter()-start))
    print(f'Videos and previews saved to {out}', flush=True)


if __name__ == '__main__': main()
