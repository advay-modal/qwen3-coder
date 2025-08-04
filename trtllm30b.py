import subprocess
import modal
import os

QWEN3_CODER_30B_MODEL_PATH = "/model_storage/qwen3-coder-30b"

app = modal.App("qwen3-coder-30b-trt-llm")
model_volume = modal.Volume.from_name("qwen3-coder-models", create_if_missing=True)

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",  # TRT-LLM requires Python 3.12
).entrypoint([]) 

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget", "clang", "libclang-dev"
).pip_install(
    "tensorrt-llm",
    "numpy",
    "pynvml",  # avoid breaking change to pynvml version API
    "flashinfer-python",
    "httpx",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
    gpu="h200",
)

def wait_for_port(process: subprocess.Popen, port: int):
    import socket

    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            if process.poll() is not None:
                raise Exception(f"Process {process.pid} exited with code {process.returncode}")

@app.cls(
    image=tensorrt_image,
    gpu="h200:4",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,  # Warm container
)
# Scale for 5 concurrent requests, allow up to 9.
@modal.concurrent(max_inputs=9, target_inputs=5)
class Model:
    
    @modal.enter()
    def enter(self):
        import httpx
        
        serve_params = {
            "host": "0.0.0.0",
            "port": 8000,
            "tp_size": 4,
        }

        serve_cmd = "trtllm-serve serve /model_storage/qwen3-coder-30b-a3b-instruct " + " ".join([f"--{k} {v}" for k, v in serve_params.items()])
        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        wait_for_port(self.serve_process, 8000)
        print("TRT-LLM server is ready!")

        self.httpx_client = httpx.Client()        

    @modal.web_server(8000)
    def serve(self):
        return

    @modal.method()
    def inference(self, json: dict, timeout: float = 4.0):
        # Adding this call-pattern - it's a faster path for requests.
        response = self.httpx_client.post(
            "http://localhost:8000/v1/chat/completions",
            json=json,
            timeout=timeout,
        )
        return response.json()

    @modal.exit()
    def exit(self):
        self.serve_process.terminate()
        print("TRT-LLM server is stopped!")