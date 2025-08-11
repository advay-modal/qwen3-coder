import subprocess
import modal
import modal.experimental

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
    "cuda-python==12.8.0",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
    gpu="h200",
)

@app.cls(
    image=tensorrt_image,
    gpu="h200:4",
    volumes={
        "/model_storage": model_volume,
    },
    min_containers=1,  # Warm container
    experimental_options={"flash": "us-east"},
)
class Model:
    
    @modal.enter()
    def enter(self):        
        serve_params = {
            "host": "0.0.0.0",
            "port": 8000,
            "tp_size": 4,
        }
        serve_cmd = "trtllm-serve serve /model_storage/qwen3-coder-30b-a3b-instruct " + " ".join([f"--{k} {v}" for k, v in serve_params.items()])
        self.serve_process = subprocess.Popen(serve_cmd, shell=True)
        self.flash_handle = modal.experimental.flash_forward(8000)

    @modal.exit()
    def exit(self):
        print("Stopping TRT-LLM server")
        self.serve_process.terminate()

        print("Stopping flash handle")
        self.flash_handle.stop()

        print("Waiting 5 seconds to finish requests")

        print("Closing flash handle")
        self.flash_handle.close()

#  curl -X POST https://modal-labs-advay-dev--qwen3-coder-30b-trt-llm-model-serve.modal.run/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       { "role": "user", "content": "Write a quick sort algorithm." }
#     ], 
#     "model": "dummy",
#     "temperature": 0.7
#   }'