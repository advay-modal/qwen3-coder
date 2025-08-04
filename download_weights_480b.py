import modal

app = modal.App("qwen3-coder-480b-a35b-instruct-download-weights")

model_volume = modal.Volume.from_name("qwen3-coder-models", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("huggingface-hub")

@app.function(image=image, volumes={"/model_storage": model_volume}, timeout=86400)
def download_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        local_dir="/model_storage/qwen3-coder-480b-a35b-instruct",
        local_dir_use_symlinks=False,
    )

@app.local_entrypoint()
def main():
    download_weights.remote()