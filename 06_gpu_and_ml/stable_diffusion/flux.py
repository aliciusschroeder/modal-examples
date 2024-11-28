# ---
# output-directory: "/tmp/flux"
# args: ["--no-compile"]
# tags: ["use-case-image-video-3d", "featured"]
# ---

# # Run Flux fast on H100s with `torch.compile`

# In this guide, we'll run Flux as fast as possible on Modal using open source tools.
# We'll use `torch.compile` and NVIDIA H100 GPUs.

# ## Setting up the image and dependencies

import time
from io import BytesIO
from pathlib import Path
import os
from PIL import Image

import modal

# We'll make use of the full [CUDA toolkit](https://modal.com/docs/guide/cuda)
# in this example, so we'll build our container image off of the `nvidia/cuda` base.

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# Now we install most of our dependencies with `apt` and `pip`.
# For Hugging Face's [Diffusers](https://github.com/huggingface/diffusers) library
# we install from GitHub source and so pin to a specific commit.

# PyTorch added [faster attention kernels for Hopper GPUs in version 2.5

diffusers_commit_sha = "8d477daed507801a50dc9f285c982b1c8051ae2d"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark",
        "transformers",
        "huggingface_hub[hf_transfer]",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "torch",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Later, we'll also use `torch.compile` to increase the speed further.
# Torch compilation needs to be re-executed when each new container starts,
# So we turn on some extra caching to reduce compile times for later containers.

flux_image = flux_image.env(
    {"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"}
).env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})

# Finally, we construct our Modal [App](https://modal.com/docs/reference/modal.App),
# set its default image to the one we just constructed,
# and import `FluxPipeline` for downloading and running Flux.1.

app = modal.App("example-flux", image=flux_image, secrets=[modal.Secret.from_name("huggingface-secret")])

with flux_image.imports():
    import torch
    from diffusers import FluxFillPipeline
    from diffusers.utils import load_image
    from PIL import Image

# ## Defining a parameterized `Model` inference class

# Next, we map the model's setup and inference code onto Modal.

# 1. We run any setup that can be persisted to disk in methods decorated with `@build`.
# In this example, that includes downloading the model weights.
# 2. We run any additional setup, like moving the model to the GPU, in methods decorated with `@enter`.
# We do our model optimizations in this step. For details, see the section on `torch.compile` below.
# 3. We run the actual inference in methods decorated with `@method`.

MAX_HEIGHT = 1024
MAX_WIDTH = 1024
MINUTES = 60  # seconds
VARIANT = "Fill-dev"  # or "dev", but note [dev] requires you to accept terms and conditions on HF


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    container_idle_timeout=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
)
class Model:
    compile: int = (  # see section on torch.compile below for details
        modal.parameter(default=0)
    )

    def setup_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        snapshot_download(f"black-forest-labs/FLUX.1-{VARIANT}", use_auth_token=os.environ["HF_TOKEN"])

        move_cache()

        pipe = FluxFillPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", 
            torch_dtype=torch.bfloat16,
        )

        return pipe

    @modal.build()
    def build(self):
        self.setup_model()

    @modal.enter()
    def enter(self):
        pipe = self.setup_model()
        pipe.to("cuda")  # move model to GPU
        self.pipe = optimize(pipe, compile=bool(self.compile))
        #self.meta_mask = load_image("https://huggingface.co/datasets/alecccdd/wmr/resolve/main/better-meta-mask.png")
        self.meta_mask = load_image("https://huggingface.co/datasets/alecccdd/wmr/resolve/main/meta-mask.png")

    @modal.method()
    def inference(self, url: str, steps: int = 25) -> bytes:
        image, mask, width, height = get_image_and_mask_and_size(url, self.meta_mask)

        print("🎨 generating image...")
        out = self.pipe(
            prompt="a woman",
            image=image,
            mask_image=mask,
            height=height,
            width=width,
            max_sequence_length=512,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()


# ## Calling our inference function

# To generate an image we just need to call the `Model`'s `generate` method
# with `.remote` appended to it.
# You can call `.generate.remote` from any Python environment that has access to your Modal credentials.
# The local environment will get back the image as bytes.

# Here, we wrap the call in a Modal [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
# so that it can be run with `modal run`:

# ```bash
# modal run flux.py
# ```

# By default, we call `generate` twice to demonstrate how much faster
# the inference is after cold start. In our tests, clients received images in about 1.2 seconds.
# We save the output bytes to a temporary file.


@app.local_entrypoint()
def main(
    url: str = "https://huggingface.co/datasets/alecccdd/wmr/resolve/main/src/00001.jpg",
    twice: bool = True,
    compile: bool = False,
):
    print("url: ", url, "is currently ignored! change later :)")
    url1 = "https://huggingface.co/datasets/alecccdd/wmr/resolve/main/src/00001.jpg"
    url2 = "https://static1.mileroticos.com/photos/d/2024/05/23/79/cb8b845379c5083a966b5099c7bbab84.jpg"
    output_path1 = Path("/tmp") / "flux" / "url2-1steps.jpg"
    output_path1.parent.mkdir(exist_ok=True, parents=True)

    stats = "Stats: \n1 step,5 steps,25 steps,40 steps,50 steps, resolution"

    def durchlauf(steps, url_nr, stats=stats):
        print("🎨 Beginning Flux inference...")
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(url2, steps=steps)
        print(f"🎨 Inference latency ({steps} step(s), url {url_nr}): {time.time() - t0:.2f} seconds")
        output_path = Path("/tmp") / "flux" / f"url{url_nr}-{steps}steps.jpg"
        print(f"🎨 saving outputs to {output_path}")
        output_path.write_bytes(image_bytes)
        return stats + f"{time.time() - t0:.2f},"

    #if twice:
        #t0 = time.time()
        #image_bytes2 = Model(compile=compile).inference.remote(url2, steps=5)
        #print(f"🎨 second inference latency (2 steps, url 2): {time.time() - t0:.2f} seconds")
    
    #TODO: clean up nach gestern. nach 25 steps wird es nicht mehr besser. Kosten, Deployment, handling, etc. planen
    # Und mehr feiern, was für einen geilen Erfolg wir heute hatten! :D

    pnr = 11
    
    stats += durchlauf(1, pnr)
    stats += durchlauf(5, pnr)
    stats += durchlauf(25, pnr)
    stats += durchlauf(40, pnr)
    stats += durchlauf(50, pnr)

    url2 = "https://static1.mileroticos.com/photos/d/2024/05/03/f0/0ce4e4fe263e7d7c37f66381285f25b8.jpg"
    pnr = pnr + 1
    stats += "\n"
    stats += durchlauf(1, pnr)
    stats += durchlauf(5, pnr)
    stats += durchlauf(25, pnr)
    stats += durchlauf(40, pnr)
    stats += durchlauf(50, pnr)
    print(stats)

# ## Speeding up Flux with `torch.compile`

# By default, we do some basic optimizations, like adjusting memory layout
# and re-expressing the attention head projections as a single matrix multiplication.
# But there are additional speedups to be had!

# PyTorch 2 added a compiler that optimizes the
# compute graphs created dynamically during PyTorch execution.
# This feature helps close the gap with the performance of static graph frameworks
# like TensorRT and TensorFlow.

# Here, we follow the suggestions from Hugging Face's
# [guide to fast diffusion inference](https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion),
# which we verified with our own internal benchmarks.
# Review that guide for detailed explanations of the choices made below.

# The resulting compiled Flux `schnell` deployment returns images to the client in under a second (~700 ms), according to our testing.
# _Super schnell_!

# Compilation takes up to twenty minutes on first iteration.
# As of time of writing in late 2024,
# the compilation artifacts cannot be fully serialized,
# so some compilation work must be re-executed every time a new container is started.
# That includes when scaling up an existing deployment or the first time a Function is invoked with `modal run`.

# We cache compilation outputs from `nvcc`, `triton`, and `inductor`,
# which can reduce compilation time by up to an order of magnitude.
# For details see [this tutorial](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

# You can turn on compilation with the `--compile` flag.
# Try it out with:

# ```bash
# modal run flux.py --compile
# ```

# The `compile` option is passed by a [`modal.parameter`](https://modal.com/docs/reference/modal.parameter#modalparameter) on our class.
# Each different choice for a `parameter` creates a [separate auto-scaling deployment](https://modal.com/docs/guide/parameterized-functions).
# That means your client can use arbitrary logic to decide whether to hit a compiled or eager endpoint.

def crop_center(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Crop the image to the specified width and height, keeping the center.

    :param image: A PIL.Image.Image instance.
    :param width: The width of the crop.
    :param height: The height of the crop.
    :return: A new PIL.Image.Image object with the cropped dimensions.
    """
    original_width, original_height = image.size

    # Calculate the cropping box
    left = (original_width - width) // 2
    top = (original_height - height) // 2
    right = left + width
    bottom = top + height

    # Perform the crop
    return image.crop((left, top, right, bottom))

def get_image_and_mask_and_size(url: str, meta_mask: Image.Image) -> tuple[Image.Image, Image.Image, int, int]:
  input = load_image(url)
  if input.size[0] > MAX_WIDTH or input.size[1] > MAX_HEIGHT:
    input = crop_center(input, min(input.size[0], MAX_WIDTH), min(input.size[1], MAX_HEIGHT))
  width = input.size[0]
  height = input.size[1]
  new_mask = crop_center(meta_mask, width, height)
  return input, new_mask, width, height

def optimize(pipe, compile=True):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compiliation (may take up to 20 minutes)...")
    mask = load_image("https://huggingface.co/datasets/alecccdd/wmr/resolve/main/mask/00001.png")
    mask = crop_center(mask, 16, 16)

    #TODO: Find out if we need to use real-world data for compilation to have real-world benefits
    pipe(
        prompt="nothing",
        image=mask,
        mask_image=mask,
        height=16,
        width=16,
        max_sequence_length=73,
        num_inference_steps=1,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe
