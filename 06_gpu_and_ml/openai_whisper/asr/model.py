import modal

from typing import Annotated
from modal import Image
import logging
import time

INFERENCE_PRECISION = "float16"
WEIGHT_ONLY_PRECISION = "int8"
MAX_BEAM_WIDTH = 4
MAX_BATCH_SIZE = 8
WHISPER_OUTPUT_DIR = f"whisper_large_v3_weights_{WEIGHT_ONLY_PRECISION}"
WHISPER_CHECKPOINT_DIR= f"whisper_large_v3_{WEIGHT_ONLY_PRECISION}"


MAX_INPUT_LEN = 256
SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN}"
DTYPE = "float16"
PLUGIN_ARGS = f"--gemm_plugin={DTYPE} --gpt_attention_plugin={DTYPE}"



N_GPUS = 1
GPU_CONFIG = modal.gpu.H100(count=N_GPUS)
DTYPE = "float16"

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
        force_build=True
    )
    .apt_install(
      "openmpi-bin",
      "libopenmpi-dev",
      "git",
      "git-lfs",
      "wget",
    )
    .run_commands([  # get rid of CUDA banner
        "rm /opt/nvidia/entrypoint.d/10-banner.sh",
        "rm /opt/nvidia/entrypoint.d/12-banner.sh",
        "rm /opt/nvidia/entrypoint.d/15-container-copyright.txt",
        "rm /opt/nvidia/entrypoint.d/30-container-license.txt",
    ])
    .run_commands([
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
        "wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav",
        "wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/convert_checkpoint.py",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/run.py",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/requirements.txt",
        "wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/tokenizer.py",
        
        "pip install -r whisper_scripts/requirements.txt"
    ])
    .pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.6",
        "requests~=2.31.0",
    )
    .run_commands(
        [
            f"python whisper_scripts/convert_checkpoint.py \
                --use_weight_only \
                --weight_only_precision {WEIGHT_ONLY_PRECISION} \
                --output_dir {WHISPER_CHECKPOINT_DIR}"
        ], gpu=GPU_CONFIG,
    )
    .run_commands(
        [
            f"trtllm-build --checkpoint_dir {WHISPER_CHECKPOINT_DIR}/encoder \
                  --output_dir {WHISPER_OUTPUT_DIR}/encoder \
                  --paged_kv_cache disable \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --use_custom_all_reduce disable \
                  --max_batch_size {MAX_BATCH_SIZE} \
                  --gemm_plugin disable \
                  --bert_attention_plugin {INFERENCE_PRECISION} \
                  --remove_input_padding disable \
                  --max_input_len 1500",
        ], gpu=GPU_CONFIG
    )
    .run_commands(
        [
            f"trtllm-build  --checkpoint_dir {WHISPER_CHECKPOINT_DIR}/decoder \
                  --output_dir {WHISPER_OUTPUT_DIR}/decoder \
                  --paged_kv_cache disable \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --use_custom_all_reduce disable \
                  --max_beam_width {MAX_BEAM_WIDTH} \
                  --max_batch_size {MAX_BATCH_SIZE} \
                  --max_seq_len 114 \
                  --max_input_len 14 \
                  --max_encoder_input_len 1500 \
                  --gemm_plugin {INFERENCE_PRECISION} \
                  --bert_attention_plugin {INFERENCE_PRECISION} \
                  --gpt_attention_plugin {INFERENCE_PRECISION} \
                  --remove_input_padding disable"
        ], gpu=GPU_CONFIG,
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
    .pip_install("pydantic==1.10.11")
    .pip_install("librosa")
    .run_commands(["wget --directory-prefix=whisper_scripts https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/whisper/whisper_utils.py"])
)

app = modal.App("faster-v2", image=image)

@app.cls(keep_warm=1, allow_concurrent_inputs=1, concurrency_limit=1, gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def enter(self):
        self.assets_dir = "/assets"
        import sys
        sys.path.append("/whisper_scripts")
        # import importlib.util

        from run import WhisperTRTLLM, decode_dataset
        self.whisper_model = WhisperTRTLLM(f"/{WHISPER_OUTPUT_DIR}", assets_dir=self.assets_dir)

        from whisper.normalizers import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        _, _ = decode_dataset(
            self.whisper_model,
            "hf-internal-testing/librispeech_asr_dummy",
            normalizer=normalizer,
            mel_filters_dir=self.assets_dir,
        )

    @modal.asgi_app()
    def web(self):
        import io
        import librosa
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import PlainTextResponse

        webapp = FastAPI()
        import sys
        sys.path.append("/whisper_scripts")
        from run import decode_wav_file

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            logger.info("health check")
            return "OK"

        @webapp.post("/")
        @webapp.post("/predict")
        async def predict(
            file: Annotated[UploadFile, File()],
        ):
            print(f"Entered predict at {time.monotonic()}")
            contents = file.file.read()
            print("Read file")
            audio_data, _ = librosa.load(io.BytesIO(contents), sr=None)
            print("Loading audio data")
            results, _ = decode_wav_file(
                audio_data,
                self.whisper_model,
                mel_filters_dir=self.assets_dir,
            )
            result_sentence = results[0][2]
            print(f"Left predict at {time.monotonic()}")
            return result_sentence
        

        return webapp