# Walkthrough video

`../gpu-observer-walkthrough.mp4` — 2:18, 1920×1080, H.264 30 fps, silent audio
track, 7.5 MB. A captioned walkthrough with no voice-over: key slides, then the
hosted viewer driven for real (load the sample trace, step 5, step 67, the GPU
node, kernel ownership, evidence tiers). There is no live-demo footage; the live
view appears as a real screenshot from the DGX Spark.

Rebuild (needs geckodriver on 127.0.0.1:4444, and `imageio-ffmpeg` + Pillow in a
Python environment):

```bash
mkdir -p /tmp/vid/raw
node submission/video/capture.mjs /tmp/vid/raw "$PWD"        # stills + click positions
python submission/video/compose.py /tmp/vid                   # captions, pointer, fades
ffmpeg -f concat -safe 0 -i /tmp/vid/list.txt -f lavfi -i anullsrc=r=48000:cl=stereo \
  -vf "fps=30,format=yuv420p" -c:v libx264 -crf 20 -c:a aac -b:a 96k -shortest \
  -movflags +faststart gpu-observer-walkthrough.mp4
```

`compose.py` expects the stills in `<dir>/raw` and writes frames to `<dir>/frames`.
