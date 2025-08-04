import subprocess
import modal
import modal.experimental

sglang_image = modal.Image.from_registry("lmsysorg/sglang:v0.4.9.post2-cu126").pip_install("httpx")

app = modal.App("qwen3-coder-480b-sglang")

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
    gpu="h100:8",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,
    timeout=86400,
)
@modal.experimental.clustered(size=2, rdma=True)
class Model:
    @modal.enter()
    def enter(self):
        import httpx
        cluster_info = modal.experimental.get_cluster_info()
        container_rank = cluster_info.rank
        first_node_hostname = "10.100.0.1"
        port = 8000

        serve_params = {
            # Read from volume with fine-tuned model weights
            "model-path": "/model_storage/qwen3-coder-480b-a35b-instruct",
            "nnodes": 2, 
            "node-rank": container_rank,
            "dist-init-addr": f"{first_node_hostname}:{port}",
            "tp": 8,
            "pp": 2,
        }
        serve_cmd = "python -m sglang.launch_server " + " ".join([f"--{k} {v}" for k, v in serve_params.items()])

        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        wait_for_port(self.serve_process, 8000)
        print("SGLang server is ready!")

        self.httpx_client = httpx.Client()

    @modal.web_server(8000)
    def serve(self):
        return

    # @modal.method()
    # def inference(self, json: dict, timeout: float = 4.0):
    #     response = self.httpx_client.post(
    #         "http://localhost:8000/v1/chat/completions",
    #         json=json,
    #         timeout=timeout,
    #     )
    #     return response.json()

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