import subprocess
import modal
import modal.experimental

sglang_image = modal.Image.from_registry("lmsysorg/sglang:v0.4.10.post2-cu126")

app = modal.App("qwen3-coder-480b-sglang")

model_volume = modal.Volume.from_name("qwen3-coder-models", create_if_missing=True)

SGLANG_PORT = 30000
MULTINODE_PORT = 5000

@app.cls(
    image=sglang_image,
    gpu="h100:8",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,
    timeout=86400,
    experimental_options={"flash": "us-east"},
)
@modal.experimental.clustered(size=2, rdma=True)
class Model:
    @modal.enter()
    def enter(self):
        cluster_info = modal.experimental.get_cluster_info()
        container_rank = cluster_info.rank

        serve_params = {
            # Read from volume with fine-tuned model weights
            "model-path": "/model_storage/qwen3-coder-480b-a35b-instruct",
            "nnodes": 2, 
            "node-rank": container_rank,
            "dist-init-addr": f"10.100.0.1:{MULTINODE_PORT}",
            "tp": 8,
            "pp": 2,
            "port": SGLANG_PORT,
            "host": "0.0.0.0",
        }
        serve_cmd = "python -m sglang.launch_server " + " ".join([f"--{k} {v}" for k, v in serve_params.items()])

        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        if container_rank == 0:
            self.flash_handle = modal.experimental.flash_forward(SGLANG_PORT)

    @modal.exit()
    def exit(self):
        print("Stopping SGLang server")
        self.serve_process.terminate()

        cluster_info = modal.experimental.get_cluster_info()
        container_rank = cluster_info.rank
        if container_rank == 0:
            print("Stopping flash handle")
            self.flash_handle.stop()

            print("Waiting 5 seconds to finish requests")

            print("Closing flash handle")
            self.flash_handle.close()

# curl -v -X POST https://modal-labs-advay-dev--qwen3-coder-480b-sglang-model.us-east.modal.direct/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       { "role": "user", "content": "Write a quick sort algorithm." }
#     ],
#     "model": "dummy",
#     "temperature": 0.7
#   }'            