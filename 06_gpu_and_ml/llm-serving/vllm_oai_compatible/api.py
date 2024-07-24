# ---
# cmd: ["modal", "deploy", "06_gpu_and_ml/llm-serving/vllm_oai_compatible/api.py", "&&", "pip", "install", "openai==1.13.3", "&&" "python", "06_gpu_and_ml/llm-serving/vllm_oai_compatible/client.py"]
# ---
# # Run an OpenAI-Compatible vLLM Server
#
# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# OpenAI's API has emerged as a standard interface for LLMs,
# and it is supported by open source LLM serving frameworks like vLLM.
#
# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.
# Note that the vLLM server is a FastAPI app, which can be configured and extended just like any other.
# Here, we use it to add simple authentication middleware.
# This implementation is based on the OpenAI-compatible server provided by vLLM.
# For the latest reference, see the vLLM documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Note: The chat template should be specified as a command-line argument
# when starting the server, e.g., --chat-template /path/to/chat_template.jinja
#
# ## Set up the container image
#
# Our first order of business is to define the environment our server will run in: the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll build it up, step-by-step, from a slim Debian Linux image.
#
# First, we install some dependencies with `pip`.

import logging
import traceback

import modal

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    [
        "vllm==0.5.3post1",  # LLM serving (exact version for compatibility)
        "huggingface_hub==0.23.2",  # download models from the Hugging Face Hub
        "hf-transfer==0.1.6",  # download models faster
        "jinja2==3.1.2",  # for chat template processing
    ]
)

# Then, we need to get hold of the weights for the model we're serving:
# Meta's LLaMA 3-8B Instruct. We create a Python function for this and add it to the image definition,
# so that we only need to download it when we define the image, not every time we run the server.
#
# If you adapt this example to run another model,
# note that for this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# the `HF_TOKEN` environment variable must be set and provided as a [Modal Secret](https://modal.com/secrets).


MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"
MODEL_DIR = f"/models/{MODEL_NAME}"


def download_model_to_image(model_dir, model_name, model_revision):
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        revision=model_revision,
    )


MINUTES = 60

vllm_image = vllm_image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
    download_model_to_image,
    timeout=20 * MINUTES,
    kwargs={
        "model_dir": MODEL_DIR,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
    },
)

# ## Build the server
#
# vLLM's OpenAI-compatible server is a [FastAPI](https://fastapi.tiangolo.com/) app.
#
# FastAPI is a Python web framework that implements the [ASGI standard](https://en.wikipedia.org/wiki/Asynchronous_Server_Gateway_Interface),
# much like [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) is a Python web framework
# that implements the [WSGI standard](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface).
#
# Modal offers [first-class support for ASGI (and WSGI) apps](https://modal.com/docs/guide/webhooks). We just need to decorate a function that returns the app
# with `@modal.asgi_app()` (or `@modal.wsgi_app()`) and then add it to the Modal app with the `app.function` decorator.
#
# The function below first imports the FastAPI app from the vLLM library, then adds some middleware. You might also add more routes here.
#
# Then, the function creates an `AsyncLLMEngine`, the core of the vLLM server. It's responsible for loading the model, running inference, and serving responses.
#
# After attaching that engine to the FastAPI app via the `api_server` module of the vLLM library, we return the FastAPI app
# so it can be served on Modal.

app = modal.App("vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to A100s or H100s, and only then increase GPU count
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret

# Default chat template for NousResearch/Meta-Llama-3-8B-Instruct
CHAT_TEMPLATE = """[INST] {% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'system' %}{{ message['content'] }}{% endif %}{% if not loop.last %}

{% endif %}{% endfor %}{% if messages[-1]['role'] != 'user' %}{{ input }}{% endif %} [/INST]"""


@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(count=N_GPU),
    container_idle_timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
@modal.asgi_app()
def serve(chat_template: str = None):
    logging.basicConfig(level=logging.INFO)
    logging.info("Entering serve function")
    logging.info(f"MODEL_DIR: {MODEL_DIR}")
    logging.info(f"N_GPU: {N_GPU}")
    logging.info(f"Chat template: {chat_template}")

    import os

    import vllm

    logging.info(f"Using vLLM version: {vllm.__version__}")

    # Use the default chat template if none is provided
    if not chat_template:
        chat_template = CHAT_TEMPLATE
        logging.info("Using default chat template")
    elif os.path.isfile(chat_template):
        with open(chat_template, "r") as f:
            chat_template = f.read()
        logging.info(f"Chat template loaded from file: {chat_template[:50]}...")
    else:
        logging.info("Chat template provided as string")

    import jinja2

    try:
        # Validate Jinja2 format
        jinja2.Template(chat_template)
        logging.info("Chat template validated as correct Jinja2 format")
    except jinja2.exceptions.TemplateError as e:
        logging.error(
            f"Error: Invalid Jinja2 format in chat template: {str(e)}"
        )
        raise
    except Exception as e:
        logging.error(f"Error loading chat template: {str(e)}")
        raise

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )

    logging.info("All necessary imports completed successfully")

    logging.info("Initializing AsyncEngineArgs")
    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="auto",  # Let vLLM automatically determine the appropriate dtype
        tensor_parallel_size=N_GPU,  # Use all available GPUs
        max_num_batched_tokens=8192,  # Maximum number of tokens to process in a batch
        max_num_seqs=256,  # Limit the number of concurrent sequences
    )
    logging.info("AsyncEngineArgs initialized with:")
    for key, value in vars(engine_args).items():
        logging.info(f"  {key}: {value}")

    try:
        logging.info("Creating AsyncLLMEngine...")
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logging.info(
            f"AsyncLLMEngine created successfully. Engine details: {engine}"
        )

        model_config = {"model": MODEL_DIR, "tokenizer": MODEL_DIR}
        served_model_names = [MODEL_DIR]
        print("Initializing OpenAIServingChat with engine and model_config")
        print(f"Model config: {model_config}")
        openai_serving_chat = OpenAIServingChat(
            engine=engine,
            model_config=model_config,
            served_model_names=served_model_names,
            chat_template=chat_template,
            response_role="assistant",
            lora_modules=[],
            prompt_adapters=[],
            request_logger=None,
        )
        logging.info("OpenAIServingChat initialized successfully")

        print(
            "Initializing OpenAIServingCompletion with engine and model_config"
        )
        print(f"Model config: {model_config}")
        openai_serving_completion = OpenAIServingCompletion(
            engine=engine,
            model_config=model_config,
            served_model_names=served_model_names,
        )
        logging.info("OpenAIServingCompletion initialized successfully")

        app = FastAPI()
        app.include_router(openai_serving_chat.router)
        app.include_router(openai_serving_completion.router)
        logging.info("Chat and Completion routers included in the app")

        # security: CORS middleware for external requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logging.info("CORS middleware added")

        # security: auth middleware
        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != f"Bearer {TOKEN}":
                return JSONResponse(
                    content={"error": "Unauthorized"}, status_code=401
                )
            return await call_next(request)

        logging.info("Authentication middleware added")

    except Exception as e:
        logging.error(f"Error initializing server: {str(e)}")
        logging.error("Stack trace: %s", traceback.format_exc())
        print(f"Error initializing server: {str(e)}")
        print("Stack trace: %s" % traceback.format_exc())
        raise

    logging.info("Server initialization completed successfully")
    return app


# ## Deploy the server
#
# To deploy the API on Modal, just run
# ```bash
# modal deploy api.py
# ```
#
# This will create a new app on Modal, build the container image for it, and deploy.
#
# ### Interact with the server
#
# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--vllm-openai-compatible-serve.modal.run`.
#
# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--vllm-openai-compatible-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output.
#
# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.
#
# To interact with the API programmatically, you can use the Python `openai` library.
#
# See the small test `client.py` script included with this example for details.
#
# ```bash
# # pip install openai==1.13.3
# python client.py
# ```
