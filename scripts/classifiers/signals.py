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

MIN_SCORE_THRESHOLD = 6
CONFIDENCE_DIVISOR = 30

TRAINING_DISCLOSURE_CAP = 0.6

EXPLICIT_TRAINING_DISCLOSURE_RE = re.compile(
    r'(?:trained?\s+on|training\s+(?:was\s+)?(?:done|performed|conducted|run|utilized)\s+on|'
    r'training\s+(?:hardware|infrastructure|utilized)|'
    r'fine[- ]?tun(?:ed|ing)\s+on|pre[- ]?train(?:ed|ing)\s+on|'
    r'we\s+train(?:ed)?\b|experiments?\b.{0,60}?\bconducted\s+on|'
    r'required\b.{0,60}?\b(?:gpu|gpus|tpu|tpus|h100|v100|a100|h200|p100|t4)|'
    r'\d+(?:\.\d+)?\s*[MBK]?\s*gpu\s+hours?)',
    re.IGNORECASE,
)

HARDWARE_LITERAL_RE = re.compile(
    r'(?:\bTPU(?:\s*v\d+)?\b|\bA100\b|\bH100\b|\bV100\b|\bH200\b|\bP100\b|\bT4\b|'
    r'\bMI\d{3}[Xx]?\b|\bGaudi\b|\bTrainium\b|\bInferentia\b|\bGPU(?:s)?\b|NVIDIA)',
    re.IGNORECASE,
)

HARDWARE_DURATION_RE = re.compile(
    r'(?:'
    # N × CHIP … training|compute|hours
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*(?:x\s*)?(?:A100|H100|V100|H200|P100|T4|MI\d{3}[Xx]?|'
    r'TPU(?:\s*v\d+)?|GPU)\b.{0,60}?\b(?:training|compute|computation|hours?)|'
    # CHIP … training|compute|hours
    r'\b(?:A100|H100|V100|H200|P100|T4|MI\d{3}[Xx]?|TPU(?:\s*v\d+)?)\b.{0,60}?'
    r'\b(?:training|compute|computation|hours?)|'
    # N gpu hours … CHIP
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*gpu\s+hours?\b.{0,80}?\b(?:A100|H100|V100|H200|P100|T4|'
    r'MI\d{3}[Xx]?|TPU(?:\s*v\d+)?)|'
    # hours (of) … CHIP  — covers "1000 hours of A100"
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*hours?\b.{0,40}?\b(?:A100|H100|V100|H200|P100|T4|'
    r'MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|GPU)|'
    # training|compute|computation (on|with|using) … CHIP  — covers "training on H100"
    r'\b(?:training|compute|computation|trained|fine[- ]?tuned|pre[- ]?trained)\b.{0,80}?'
    r'\b(?:A100|H100|V100|H200|P100|T4|MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|Gaudi|Trainium)\b'
    r')',
    re.IGNORECASE,
)


def _snippet_text(snippet):
    if isinstance(snippet, dict):
        return snippet.get("snippet", "") or ""
    return str(snippet) if snippet else ""


# Conditional/hypothetical phrasing that turns a training-disclosure phrase into
# a suggestion rather than a fact (e.g. "can be fine-tuned on H100" — users could
# do this, not that it was done). Reject such snippets from the cap lift.
_CONDITIONAL_DISCLOSURE_RE = re.compile(
    r'\b(?:can|could|may|might|should|would|will|recommended|allow[s]?|able\s+to|capable\s+of)\b'
    r'[^.]{0,40}?\b(?:be\s+)?(?:trained?|fine[- ]?tuned|pre[- ]?trained|run|deployed)\b',
    re.IGNORECASE,
)


def snippet_is_training_disclosure(snippet):
    """Does this single snippet co-locate a training phrase with a hardware literal,
    or contain a hardware+duration phrase? Rejects conditional/hypothetical phrasing.
    (Strict — the chip-confidence cap and aggregation policy rely on this.)"""
    text = _snippet_text(snippet)
    if not text:
        return False
    if _CONDITIONAL_DISCLOSURE_RE.search(text):
        return False
    if EXPLICIT_TRAINING_DISCLOSURE_RE.search(text) and HARDWARE_LITERAL_RE.search(text):
        return True
    if HARDWARE_DURATION_RE.search(text):
        return True
    return False


def snippet_is_training_context(snippet):
    """Looser check: contains a non-hypothetical training-disclosure phrase or
    a training-launcher invocation (torchrun/CUDA_VISIBLE_DEVICES/accelerate
    launch/deepspeed), regardless of whether a specific chip is named. Used
    to surface candidate quotes to the LLM — the LLM decides whether the
    context is about *this* model's training vs a dataset / user fine-tuning."""
    text = _snippet_text(snippet)
    if not text:
        return False
    if _CONDITIONAL_DISCLOSURE_RE.search(text):
        return False
    if EXPLICIT_TRAINING_DISCLOSURE_RE.search(text):
        return True
    # Training-launcher invocations (see _TRAINING_LAUNCHER_RE below) — defined
    # later in the file, so defer the reference at call time.
    return bool(_TRAINING_LAUNCHER_RE.search(text))


# Training-launcher / distributed-training invocations. These are how training
# scripts (train.py, run_train.sh, accelerate_config.yaml, etc.) manifest
# hardware usage in code — there's no natural-language "trained on H100" but
# there IS `torchrun --nproc-per-node=8` or `CUDA_VISIBLE_DEVICES=0,1,2,3`.
_TRAINING_LAUNCHER_RE = re.compile(
    r'(?:'
    # Shell launchers
    r'torchrun\s+[^\n]*--nproc[-_]?per[-_]?node|'
    r'python\s+-m\s+torch\.distributed|'
    r'accelerate\s+launch|'
    r'deepspeed\s+(?:--|\w+\.py)|'
    r'\bCUDA_VISIBLE_DEVICES\s*=\s*[\d,]+|'
    r'--num[-_]?gpus?[=\s]+\d+|'
    r'--num[-_]?nodes?[=\s]+\d+|'
    r'--nproc[-_]?per[-_]?node|'
    # Python-level distributed training setup
    r'mp\.spawn\s*\(|'
    r'DistributedDataParallel|'
    r'torch\.distributed\.(?:launch|init_process_group|run)|'
    r'import\s+torch\.distributed|'
    r'from\s+torch\.distributed|'
    r'from\s+accelerate\s+import|'
    r'\bAccelerator\s*\(|'
    r'deepspeed\.init_distributed|'
    r'init_process_group|'
    r'ddp_find_unused_parameters|'
    r'\bnccl\b|'
    r'--tensor[-_]?parallel|'
    r'--pipeline[-_]?parallel|'
    # Explicit CUDA training ops (distinct from runtime "device = cuda:0")
    r'\.cuda\(\)\s*$|'  # ".cuda()" at end of line is usually model .cuda() in training
    r'model\.cuda\(\)|'
    # TPU / XLA training
    r'import\s+jax|xm\.optimizer_step|xm\.xla_device|'
    r'torch_xla\.core|flax\.training'
    r')',
    re.IGNORECASE,
)


# A file path that looks like training code (Python train scripts, bash
# launchers under scripts/, accelerate / deepspeed configs). Docs/READMEs
# don't count — we want the launcher to sit in code that actually runs.
_TRAINING_FILE_PATH_RE = re.compile(
    r'(?:^|/)(?:'
    r'(?:train|finetune|pretrain|run_train|run_finetune|run_pretrain)[^/]*\.(?:py|sh)|'
    r'scripts?/[^/]*\.(?:sh|py)|'
    r'accelerate[^/]*\.ya?ml|'
    r'ds_config[^/]*\.json|'
    r'deepspeed[^/]*\.json'
    r')$',
    re.IGNORECASE,
)


_LAUNCHER_CHIP_CLASSIFIER = [
    # (regex on snippet text, implied chip)
    (re.compile(r'\b(?:jax\.distributed|TPUStrategy|torch_xla\.core|flax\.training|xm\.(?:optimizer_step|xla_device))\b', re.IGNORECASE), "google_tpu"),
    (re.compile(r'\b(?:rocm|hipify|rccl|MI\d{3}[Xx]?)\b', re.IGNORECASE), "amd"),
    # Default: CUDA / PyTorch distributed / DeepSpeed / Accelerate → nvidia
    (re.compile(
        r'torchrun|CUDA_VISIBLE_DEVICES|torch\.distributed|DistributedDataParallel|'
        r'accelerate\s+launch|from\s+accelerate\s+import|\bAccelerator\s*\(|'
        r'deepspeed|\bnccl\b|init_process_group|ddp_find_unused_parameters|'
        r'--nproc[-_]?per[-_]?node|model\.cuda\(\)',
        re.IGNORECASE), "nvidia"),
]


def launcher_implied_chip(training_snippets):
    """If any `training_snippet` has a launcher pattern AND sits in a training-code
    file (train.py, scripts/*.sh, accelerate_config.yaml, ...), return the chip
    that the launcher implies (nvidia / google_tpu / amd). Otherwise None.

    This lets us short-circuit the LLM in cases where github training scripts
    clearly encode the chip family even though no chip literal (A100/H100) is
    named — Gemma tends to over-reject these."""
    if not training_snippets:
        return None
    # TPU/AMD evidence wins over NVIDIA when both appear (explicit non-CUDA stacks).
    tpu_hit = amd_hit = nv_hit = False
    for s in training_snippets:
        src = s.get("source", "") or s.get("file", "") or s.get("section", "")
        if not _TRAINING_FILE_PATH_RE.search(src or ""):
            continue
        text = s.get("snippet", "") if isinstance(s, dict) else str(s)
        for rx, chip in _LAUNCHER_CHIP_CLASSIFIER:
            if rx.search(text):
                if chip == "google_tpu":
                    tpu_hit = True
                elif chip == "amd":
                    amd_hit = True
                elif chip == "nvidia":
                    nv_hit = True
                break
    if tpu_hit:
        return "google_tpu"
    if amd_hit:
        return "amd"
    if nv_hit:
        return "nvidia"
    return None


def extract_training_snippets(content, source="", max_snippets=4,
                              left=140, right=200):
    """Scan `content` for training-disclosure sentences that aren't hypothetical
    fine-tuning instructions, plus training-launcher invocations in scripts.
    Returns a list of {"snippet", "source"} dicts suitable for the LLM.

    Handles two patterns of disclosure:
    1. English prose ("we trained on N nodes …") — via EXPLICIT_TRAINING_DISCLOSURE_RE.
    2. Training-script code (`torchrun --nproc-per-node=8`, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
       `accelerate launch`, `deepspeed train.py`) — via TRAINING_LAUNCHER_RE.
    The LLM decides whether the context is about *this* model's training vs a
    dataset mention or a user fine-tuning suggestion."""
    if not content:
        return []
    out = []
    seen = set()

    def _emit(m, left_, right_):
        start = max(0, m.start() - left_)
        end = min(len(content), m.end() + right_)
        raw = content[start:end].replace("\n", " ")
        raw = re.sub(r'\s+', ' ', raw).strip()
        if len(raw) < 40:
            return
        key = raw[:160]
        if key in seen:
            return
        candidate = {"snippet": f"...{raw}...", "source": source}
        # Prose: must pass training-context check (rejects hypothetical);
        # launcher matches bypass that check — they're code, not prose.
        if not snippet_is_training_context(candidate) and \
                not _TRAINING_LAUNCHER_RE.search(candidate["snippet"]):
            return
        seen.add(key)
        out.append(candidate)

    for m in EXPLICIT_TRAINING_DISCLOSURE_RE.finditer(content):
        _emit(m, left, right)
        if len(out) >= max_snippets:
            return out

    for m in _TRAINING_LAUNCHER_RE.finditer(content):
        _emit(m, left, right)
        if len(out) >= max_snippets:
            return out

    return out


def has_explicit_training_chip_evidence(snippets):
    """Any snippet qualifies as a training disclosure."""
    return any(snippet_is_training_disclosure(s) for s in (snippets or []))


# Back-compat alias; identical semantics to the stricter check above.
has_training_disclosure_language = has_explicit_training_chip_evidence


def apply_training_disclosure_cap(confidence, snippets, cap=TRAINING_DISCLOSURE_CAP):
    """Cap confidence at `cap` unless a snippet provides co-located training-hardware evidence."""
    if confidence <= cap:
        return confidence
    if has_explicit_training_chip_evidence(snippets):
        return confidence
    return cap
