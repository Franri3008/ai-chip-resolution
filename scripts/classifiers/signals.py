import re

HARDWARE_SIGNALS = {
    "nvidia": {
        "strong": [
            r'\bcuda\b',
            r'torch\.cuda\b',
            r'\.cuda\(\)',
            r'\bnccl\b',
            r'\btensorrt\b',
            r'\bcudnn\b',
            r'\bnvlink\b',
            r'\bnvidia-smi\b',
            r'nvidia/cuda',
            r'\bcupy\b',
            r'\bflash[-_]?att(?:ention|n)\b',
            r'\bnvidia[-_]apex\b|from apex\b',
            r'\btriton\b',
            r'\bcutlass\b',
        ],
        "medium": [
            r'\bA100\b',
            r'\bH100\b',
            r'\bH200\b',
            r'\bV100\b',
            r'\bL40[Ss]?\b',
            r'\bA6000\b',
            r'\bA10[Gg]?\b',
            r'\bDGX\b',
            r'\bdeepspeed\b',
            r'torch\+cu\d+',
        ],
        "weak": [],
        "file_presence": [
            r'\.cu$',
            r'\.cuh$',
        ],
    },
    "amd": {
        "strong": [
            r'\brocm\b',
            r'\bhip\b(?!.*hip[-_]?hop)',
            r'\brocm-smi\b',
            r'pytorch.*rocm|rocm.*pytorch',
            r'\bhipify\b',
            r'\brocblas\b',
            r'\bmiopen\b',
            r'\brccl\b',
        ],
        "medium": [
            r'\bMI250\b',
            r'\bMI300[Xx]?\b',
            r'\bMI210\b',
            r'\binstinct\b',
            r'amd[-_]?gpu',
        ],
        "weak": [],
        "file_presence": [
            r'\.hip$',
        ],
    },
    "intel": {
        "strong": [
            r'\bopenvino\b',
            r'\boneapi\b|one[-_]?api',
            r'\bipex\b',
            r'intel[-_]extension[-_]for[-_]pytorch',
            r'\bgaudi\b',
            r'\bhabana\b',
            r'habanalabs',
            r'\bintel[-_]?neural[-_]?compressor\b',
        ],
        "medium": [
            r'\bxeon\b',
            r'\bmkl\b',
            r'\bonednn\b|one[-_]?dnn',
        ],
        "weak": [],
        "file_presence": [],
    },
    "google_tpu": {
        "strong": [
            r'\btpu\b',
            r'\bjax\b',
            r'\bflax\b',
            r'\bxla\b',
            r'tpu[-_]?strategy',
            r'\blibtpu\b',
            r'cloud[-_]?tpu',
            r'torch[-_]?xla\b|torch_xla',
            r'\boptax\b',
            r'tpu[-_]?v[2-5]',
        ],
        "medium": [
            r'\btf\.distribute\b',
            r'jaxlib',
        ],
        "weak": [],
        "file_presence": [],
    },
    "apple": {
        "strong": [
            r'\bmlx\b',
            r'\bcoreml\b|core[-_]?ml',
            r'\bmetal\b',
            r'\bmps\b',
            r'apple[-_]?silicon',
            r'\banekit\b',
            r'coremltools',
        ],
        "medium": [
            r'\bm[1-4][-_ ](?:pro|max|ultra)\b',
        ],
        "weak": [],
        "file_presence": [],
    },
    "aws": {
        "strong": [
            r'\binferentia\b',
            r'\btrainium\b',
            r'\bneuron[-_]?sdk\b|aws[-_]?neuron',
            r'torch[-_]?neuronx?',
            r'\bneuronx[-_]cc\b',
        ],
        "medium": [
            r'\btrn1\b',
            r'\binf2\b',
        ],
        "weak": [],
        "file_presence": [],
    },
    "qualcomm": {
        "strong": [
            r'\bqnn\b',
            r'\bsnpe\b',
            r'qualcomm[-_]?ai',
            r'\bsnapdragon\b',
            r'\bhexagon\b',
        ],
        "medium": [],
        "weak": [],
        "file_presence": [],
    },
}


FRAMEWORK_SIGNALS = {
    "pytorch": {
        "strong": [
            r'\bimport torch\b',
            r'\bfrom torch\b',
            r'\btorch\.nn\b',
            r'\btorch\.optim\b',
            r'\bpytorch\b',
            r'\btorch\.Tensor\b',
            r'\btorch\.utils\b',
        ],
        "medium": [
            r'\btorchvision\b',
            r'\btorchaudio\b',
            r'\btorch\.distributed\b',
        ],
        "weak": [],
    },
    "tensorflow": {
        "strong": [
            r'\bimport tensorflow\b',
            r'\bfrom tensorflow\b',
            r'\btf\.keras\b',
            r'\btensorflow\b',
            r'\btf\.data\b',
            r'\btf\.train\b',
        ],
        "medium": [
            r'\bkeras\b',
            r'\btf\.distribute\b',
        ],
        "weak": [],
    },
    "jax": {
        "strong": [
            r'\bimport jax\b',
            r'\bfrom jax\b',
            r'\bjax\.numpy\b',
            r'\bjaxlib\b',
        ],
        "medium": [
            r'\bflax\b',
            r'\boptax\b',
        ],
        "weak": [],
    },
    "paddlepaddle": {
        "strong": [
            r'\bimport paddle\b',
            r'\bfrom paddle\b',
            r'\bpaddlepaddle\b',
            r'\bpaddle\.nn\b',
        ],
        "medium": [
            r'\bpaddlenlp\b',
        ],
        "weak": [],
    },
    "mxnet": {
        "strong": [
            r'\bimport mxnet\b',
            r'\bfrom mxnet\b',
        ],
        "medium": [
            r'\bgluon(?!ts)\b(?!.*pytorch)',
        ],
        "weak": [],
    },
    "onnx": {
        "strong": [
            r'\bonnxruntime\b',
            r'\bimport onnx\b',
            r'\bfrom onnx\b',
            r'\bonnx_model\b',
        ],
        "medium": [
            r'\b\.onnx\b',
            r'\bonnx\b',
        ],
        "weak": [],
    },
}


DEPENDENCY_SIGNALS = {
    # Chip providers
    "nvidia": ["cupy", "nvidia-apex", "flash-attn", "triton", "tensorrt",
               "pynvml", "nvidia-ml-py", "cutlass"],
    "amd": ["torch-rocm", "rocm"],
    "intel": ["intel-extension-for-pytorch", "openvino", "neural-compressor",
              "habana-frameworks", "optimum-habana", "optimum-intel"],
    "google_tpu": ["jax", "jaxlib", "flax", "optax", "libtpu", "torch-xla",
                   "tensorflow-tpu"],
    "apple": ["mlx", "coremltools"],
    "aws": ["torch-neuronx", "neuronx-cc", "aws-neuron-runtime"],
    "qualcomm": ["qnn", "snpe"],
    # Frameworks
    "pytorch": ["torch", "torchvision", "torchaudio", "pytorch-lightning",
                "lightning"],
    "tensorflow": ["tensorflow", "tensorflow-gpu", "tf-nightly", "keras"],
    "jax": ["jax", "jaxlib"],
    "paddlepaddle": ["paddlepaddle", "paddlepaddle-gpu", "paddlenlp"],
    "mxnet": ["mxnet-cu", "mxnet==", "mxnet>", "mxnet<"],
    "onnx": ["onnxruntime", "onnxruntime-gpu", "onnx"],
}

CHIP_PROVIDERS = set(HARDWARE_SIGNALS.keys())
FRAMEWORKS = set(FRAMEWORK_SIGNALS.keys())

MIN_SCORE_THRESHOLD = 8
CONFIDENCE_DIVISOR = 30

TRAINING_DISCLOSURE_CAP = 0.6

EXPLICIT_TRAINING_DISCLOSURE_RE = re.compile(
    r'(?:trained?\s+on|training\s+(?:was\s+)?(?:done|performed|conducted|run)\s+on|'
    r'training\s+hardware|training\s+infrastructure|'
    r'fine[- ]?tun(?:ed|ing)\s+on|pre[- ]?train(?:ed|ing)\s+on|'
    r'we\s+train(?:ed)?\b|experiments?\b.{0,60}?\bconducted\s+on|'
    r'required\b.{0,60}?\b(?:gpu|gpus|tpu|tpus|h100|v100|a100|h200|p100|t4))',
    re.IGNORECASE,
)


def has_training_disclosure_language(snippets):
    """Return True if any snippet contains explicit training-disclosure phrasing."""
    for snippet in snippets or []:
        text = snippet.get("snippet", "") if isinstance(snippet, dict) else str(snippet)
        if text and EXPLICIT_TRAINING_DISCLOSURE_RE.search(text):
            return True
    return False


def apply_training_disclosure_cap(confidence, snippets, cap=TRAINING_DISCLOSURE_CAP):
    """Cap confidence when no snippet shows explicit training-disclosure language."""
    if confidence <= cap:
        return confidence
    if has_training_disclosure_language(snippets):
        return confidence
    return cap
