"""
SmartQC — Qualcomm AI Hub compile job
 
Submits smartqc_model.onnx to Qualcomm AI Hub to compile/optimize it for
a Snapdragon laptop target (Snapdragon X Elite), then downloads the
optimized model back into model/.
 
Run (from the SmartQC project root, with venv active, after
`qai-hub configure --api_token ...`):
    python aihub_compile.py
 
Docs: https://app.aihub.qualcomm.com/docs/
"""
 
from pathlib import Path
import time
 
import qai_hub as hub
 
ONNX_PATH = Path("model/smartqc_model.onnx")
OPTIMIZED_OUTPUT_DIR = Path("model/optimized")
OPTIMIZED_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
 
 
def main():
    print("Looking up device: 'Snapdragon X Elite CRD'")
    devices = hub.get_devices(name="Snapdragon X Elite CRD")
    for d in devices:
        print(f"  - {d.name}  (os: {d.os})")
 
    if not devices:
        raise RuntimeError(
            "Could not find 'Snapdragon X Elite CRD' on AI Hub. "
            "Run list_devices.py or find_windows_device.py to confirm the exact name."
        )
 
    target_device = devices[0]
    print(f"\nSubmitting compile job for device: {target_device.name}")
 
    compile_job = hub.submit_compile_job(
        model=str(ONNX_PATH),
        device=target_device,
        options="--target_runtime onnx",  # keeps ONNX Runtime + QNN EP compatible output
    )
 
    print(f"Compile job submitted: {compile_job.job_id}")
    print(f"Track progress at: {compile_job.url}")
    print("Waiting for the job to finish (this can take a few minutes)...")
 
    compile_job.wait()
    status = compile_job.get_status()
    print(f"Job status: {status}")
 
    if not status.success:
        raise RuntimeError(
            f"Compile job did not succeed. Check details at: {compile_job.url}"
        )
 
    # The job can report success slightly before the artifact is fully
    # available for download server-side — retry with backoff.
    optimized_model = compile_job.get_target_model()
    out_path = OPTIMIZED_OUTPUT_DIR / "smartqc_model_optimized.onnx"
 
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            optimized_model.download(str(out_path))
            break
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_s = 15 * attempt
            print(f"Download not ready yet ({e}). Retrying in {wait_s}s... "
                  f"(attempt {attempt}/{max_attempts})")
            time.sleep(wait_s)
 
    print(f"\nOptimized model downloaded to: {out_path.resolve()}")
    print("Copy/rename this file to app/model/smartqc_model.onnx to use it in the app.")
 
 
if __name__ == "__main__":
    main()
 