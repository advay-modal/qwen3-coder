import subprocess
import modal
import modal.experimental

sglang_image = modal.Image.from_registry("lmsysorg/sglang:v0.4.9.post2-cu126")

app = modal.App("qwen3-coder-30b-sglang")

model_volume = modal.Volume.from_name("qwen3-coder-models", create_if_missing=True)

@app.cls(
    image=sglang_image,
    gpu="h200:4",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,
    experimental_options={"flash": "us-east"},
)
class Model:
    @modal.enter()
    def enter(self):

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
        self.flash_handle = modal.experimental.flash_forward(8000)

    @modal.exit()
    def exit(self):
        print("Stopping SGLang server")
        self.serve_process.terminate()

        print("Stopping flash handle")
        self.flash_handle.stop()

        print("Waiting 5 seconds to finish requests")

        print("Closing flash handle")
        self.flash_handle.close()


#  curl -X POST https://modal-labs-advay-dev--qwen3-coder-30b-sglang-model-serve.modal.run/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       { "role": "user", "content": "Write a quick sort algorithm." }
#     ], 
#     "model": "dummy",
#     "temperature": 0.7
#   }'