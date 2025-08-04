import subprocess
import modal

sglang_image = modal.Image.from_registry("lmsysorg/sglang:v0.4.9.post2-cu126").pip_install("httpx")

app = modal.App("qwen3-coder-30b-sglang")

model_volume = modal.Volume.from_name("qwen3-coder-models", create_if_missing=True)

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
    image=sglang_image,
    gpu="h200:4",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,
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
            # Read from volume with fine-tuned model weights
            "model": "/model_storage/qwen3-coder-30b-a3b-instruct",
            "mem-fraction": 0.7,
            # Compile CUDA graph for decoding up to 12 concurrent requests.
            "cuda-graph-bs": " ".join(map(str, range(1, 13))),
            "tp": 4,
        }
        serve_cmd = "python -m sglang.launch_server " + " ".join([f"--{k} {v}" for k, v in serve_params.items()])

        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        wait_for_port(self.serve_process, 8000)
        print("SGLang server is ready!")

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
        print("SGLang server is stopped!")


#  curl -X POST https://modal-labs-advay-dev--qwen3-coder-30b-sglang-model-serve.modal.run/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       { "role": "user", "content": "Write a quick sort algorithm." }
#     ], 
#     "model": "dummy",
#     "temperature": 0.7
#   }'