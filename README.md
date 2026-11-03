# Loop Live

Live close-call warning from any camera pointed at a road. A YOLO11n detector runs in the browser
(ONNX Runtime Web: WebGPU, else CPU), tracks each road user, projects its path about 2 s ahead and
turns a pair red when their footprints would meet within 1 s while closing faster than ~15 km/h.
Nothing is uploaded: the video stays on the device.

- Phone: open the page, allow the camera, point it at the road from a still spot.
- Laptop + DroidCam: start DroidCam on phone and laptop, pick "DroidCam" in the camera list.
- No road nearby: "Use a video file".
- `?cpu` forces the CPU detector (for devices whose WebGPU misbehaves).

Static site: serve this folder over HTTPS (GitHub Pages works). `sw.js` adds the headers that let
the CPU detector use several threads.

Detector: Ultralytics YOLO11n (AGPL-3.0, https://github.com/ultralytics/ultralytics), exported to ONNX.
Runtime: onnxruntime-web 1.20.1 (MIT).
Rough by design: one uncalibrated camera, so distances and speeds are estimates and some warnings
are false. It predicts the next 1-2 s of motion, not crashes.
