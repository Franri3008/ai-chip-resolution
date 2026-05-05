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
            r'\bA800\b',                          # export-restricted A100 variant for China
            r'\bA40\b',                           # data-center Ampere
            r'\bH100\b',
            r'\bH800\b',                          # export-restricted H100 variant for China
            r'\bH200\b',
            r'\bV100\b',
            r'\bL40[Ss]?\b',
            r'\bA6000\b',
            r'\bA10[Gg]?\b',
            r'\bRTX\s*(?:30|40|50)\d{2}\b',       # consumer GPUs: RTX 3090, 4090, etc.
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
    "huawei_ascend": {
        # Bare "NPU" is excluded — Apple/Qualcomm also use the term.
        "strong": [
            r'\bascend\b',
            r'\bmindspore\b',
            r'\bcann\b',
            r'\bhccl\b',                          # Huawei Collective Comm Lib (NCCL analog)
            r'\bnpu[-_]?smi\b',                   # nvidia-smi analog
            r'vllm[-_]ascend',
            r'\bmindformers\b',
            r'\bmindyolo\b',
            r'\b昇腾\b',                          # Chinese for Ascend
            r'ASCEND_RT_VISIBLE_DEVICES',
            r'mindspore\.context',
            r'mindspore\.set_context',
            r'device_target\s*=\s*["\']?Ascend',
            r'/dev/davinci\d?',
            # bare `davinci` collides with OpenAI GPT-3 DaVinci; require context
            r'\bdavinci\b\s+(?:NPU|core|architecture|AI)',
            r'(?:Huawei|Ascend)\s+\w*\s*davinci',
        ],
        "medium": [
            r'\bAscend\s*\d{3}[A-Da-d]?\b',
            r'\b910[ABCDabcd]\b',                 # 910A/B/C/D — bare "910" alone too risky
            r'\bAtlas\s*[28]00\s*[ABTI]?\s*A\d?\b',  # Atlas 200I A2, 800T A2, 200T A2
            r'\bAtlas\s*9\d{2}\b',                # Atlas 900, 910 (clusters)
        ],
        "weak": [],
        "file_presence": [],
    },
    "cambricon": {
        "strong": [
            r'\bcambricon\b',
            r'\bcnml\b',                          # Cambricon ML
            r'\bcnnl\b',                          # Cambricon NN Lib (cuDNN analog)
            r'\bcndrv\b',                         # Cambricon driver
            r'\bbangpy\b',                        # BANG Python API
            r'\bMLUDevice\b',
            r'cambricon[-_]pytorch',
            r'\btorch[-_]?mlu\b',
        ],
        "medium": [
            r'\bMLU\s*\d{3}\b',                   # MLU370, MLU590
        ],
        "weak": [],
        "file_presence": [],
    },
    "baidu_kunlun": {
        # Bare "Kunlun" collides with the Chinese given name (e.g. "Kunlun Zhu"
        # in citations) and Beijing Kunlun Tech / Skywork's parent "Kunlun Inc."
        # Bare "XPU" collides with Intel oneAPI. Both require chip-noun or
        # training-verb context — same rule we already apply to NPU/MUSA/davinci.
        "strong": [
            r'\bkunlunxin\b',                     # brand, unambiguous
            # "Kunlun" followed by a chip noun: "Kunlun P800 chips", "Kunlun XPU".
            r'\bkunlun\b(?=\s+(?:P\d{3}|R[23]00|XPU|chip|core|card|cluster|'
            r'GPU|accelerat|NPU))',
            # Phrase-level disclosure: "Baidu Kunlun", "trained on Kunlun".
            r'(?:\bbaidu\s+|trained?\s+on\s+|using\s+|with\s+)(?:\d+\s+)?kunlun\b',
            r'\b昆仑(?:芯|核)?\b',
            r'\bxpurt\b',
            r'paddle[-_]?xpu|xpu[-_]?paddle',
            r'paddlepaddle[-_]xpu',
            r'paddle\.set_device\s*\(\s*["\']xpu',
            r'device\s*=\s*["\']xpu(?::\d+)?["\']',
            r'XPU_VISIBLE_DEVICES',
            r'\bbaidu[-_]?xpu\b',
            r'KUNLUN_VISIBLE_DEVICES',
        ],
        "medium": [
            r'\bP800\b',                          # Kunlun P800 (training)
            r'\bR200\b|\bR300\b',                 # Kunlun inference cards
            r'(?:Kunlun|Baidu)\s+(?:P|R)\d{3}',
        ],
        "weak": [],
        "file_presence": [],
    },
    "moore_threads": {
        "strong": [
            r'\bmusa\b(?!\s+(?:university|state|today))',
            r'\bmthreads\b',
            r'\bmoore[-_ ]?threads\b',
            r'\btorch[-_]?musa\b',
            r'vllm[-_]musa',
            r'\bmusart\b',                        # MUSA runtime (CUDA-runtime analog)
            r'\bmusatoolkit\b',
            r'\bmccl\b',                          # Moore Threads collective (NCCL analog)
            r'MUSA_VISIBLE_DEVICES',
            r'\bmtgpu\b',
        ],
        "medium": [
            r'\bMTT\s*S\d{3,4}\b',                # MTT S4000, S3000, S80
            r'(?:MTT|Moore\s*Threads)\s+S\d{3,4}',
        ],
        "weak": [],
        "file_presence": [],
    },
    "iluvatar": {
        "strong": [
            r'\biluvatar\b',
            r'\b天数智芯\b',
            r'\b天数\s*智芯\b',
            r'\bixrt\b',                          # Iluvatar inference RT
            r'\bcorex\b',                         # CoreX SDK
            r'iluvatar[-_]corex',
            r'\bixsmi\b',
            r'\bilu(?:vatar)?[-_]?gpu\b',
        ],
        "medium": [
            r'\bBI[-_ ]?V?1\d{2}\b',              # BI-V100, BI-V150
            r'\bMR[-_ ]?V?\d{3}\b',               # MR-V100
            r'(?:Iluvatar|天数)\s+\w*\s*(?:BI|MR)[-_]?V?\d{3}',
        ],
        "weak": [],
        "file_presence": [],
    },
    "hygon": {
        # Bare `DCU` is generic; require Hygon/DTK/hy-smi context.
        "strong": [
            r'\bhygon\b',
            r'\b海光\b',                          # Chinese for Hygon
            r'\bhy[-_]?smi\b',
            r'\bhygon[-_]?dcu\b',
            r'\bdtk\b(?!.*toolkit\s*for\s*(?:vmware|swift))',  # Hygon DTK; weak filter
            r'hygon[-_]?dtk',
            r'(?:Hygon|海光)\s+DCU',
            r'DCU_VISIBLE_DEVICES',
        ],
        "medium": [
            r'\bDCU\b(?=.{0,80}?(?:Hygon|海光|DTK|hy-smi|MI\d{3}|rocm))',
            r'\b[KZ]100\s*(?:DCU|AI)\b',
        ],
        "weak": [],
        "file_presence": [],
    },
    "metax": {
        "strong": [
            r'\bmetax\b',
            r'\bmuxi\b',
            r'\b沐曦\b',
            r'\bmxmaca\b',
            r'\bmx[-_]?smi\b',
            r'\bmaca\b(?!.*sequence|.*record)',   # MetaX MACA stack; filter unrelated
            r'metax[-_]?gpu|muxi[-_]?gpu',
            r'METAX_VISIBLE_DEVICES',
        ],
        "medium": [
            r'\bC[5-6]\d{2}\b(?=.{0,80}?(?:MetaX|Muxi|沐曦|MXMACA))',
            r'(?:MetaX|Muxi|沐曦)\s+C\d{3}',
        ],
        "weak": [],
        "file_presence": [],
    },
}


DEPENDENCY_SIGNALS = {
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
    "huawei_ascend": ["mindspore", "mindformers", "mindyolo", "torch-npu",
                      "ascend-cann-toolkit", "vllm-ascend"],
    "cambricon": ["torch-mlu", "cambricon-pytorch", "bangpy"],
    "baidu_kunlun": ["paddlepaddle-xpu", "xpurt", "kunlunxin"],
    "moore_threads": ["torch-musa", "vllm-musa", "musatoolkit"],
    "iluvatar": ["ixrt", "iluvatar-corex"],
    "hygon": ["hygon-dtk", "torch-dcu"],
    "metax": ["mxmaca", "torch-metax"],
}

CHIP_PROVIDERS = set(HARDWARE_SIGNALS.keys())

MIN_SCORE_THRESHOLD = 6
CONFIDENCE_DIVISOR = 30

TRAINING_DISCLOSURE_CAP = 0.6

EXPLICIT_TRAINING_DISCLOSURE_RE = re.compile(
    r'(?:trained?\s+on|training\s+(?:was\s+)?(?:done|performed|conducted|run|utilized)\s+on|'
    r'training\s+(?:hardware|infrastructure|utilized)|'
    r'fine[- ]?tun(?:ed|ing)\s+on|pre[- ]?train(?:ed|ing)\s+on|'
    r'we\s+train(?:ed)?\b|experiments?\b.{0,60}?\bconducted\s+on|'
    r'required\b.{0,60}?\b(?:gpu|gpus|tpu|tpus|h100|v100|a100|h200|p100|t4|ascend|npu|mlu)|'
    r'\d+(?:\.\d+)?\s*[MBK]?\s*(?:gpu|npu)\s+hours?|'
    # Chinese disclosure phrasing common in TeleChat / Pangu cards
    r'国产算力(?:训练)?|完全基于国产|domestic\s+(?:chinese\s+)?computing|'
    r'国产(?:化|的|深度学习)?(?:框架|算力)(?:适配)?|训练服务器|'
    # 训练 (Chinese for "training") used as a verb / heading marker.
    r'训练\s*[:：]|训练\s+(?:在|于|使用)|'
    # Trained-on-Ascend phrasings (model cards often skip "GPU" suffix for NPUs)
    r'trained?\s+(?:from\s+scratch\s+)?(?:on|based\s+on|using|with)\s+(?:the\s+)?(?:ascend|atlas|mindspore|npu|mlu|cambricon|kunlun|kunlunxin|p800|xpu|musa|s4000|mthreads|moore[-_ ]?threads|iluvatar|bi[-_ ]?v?1\d{2}|corex|hygon|dcu|dtk|metax|muxi|mxmaca))',
    re.IGNORECASE,
)

HARDWARE_LITERAL_RE = re.compile(
    r'(?:\bTPU(?:\s*v\d+)?\b|\bA100\b|\bH100\b|\bV100\b|\bH200\b|\bP100\b|\bT4\b|'
    r'\bMI\d{3}[Xx]?\b|\bGaudi\b|\bTrainium\b|\bInferentia\b|\bGPU(?:s)?\b|NVIDIA|'
    r'\bAscend(?:\s*\d{3}[A-Da-d]?)?\b|\b910[ABCDabcd]\b|\bAtlas\s*\d{3}\b|\bNPU(?:s)?\b|'
    r'\bMLU\s*\d{3}\b|\bMindSpore\b|\bCambricon\b|\b昇腾\b|'
    r'\bKunlun(?:xin)?\b|\bP800\b|\bR[23]00\b|\b昆仑(?:芯)?\b|'
    r'\bMUSA\b|\bMTT\s*S\d{3,4}\b|\bMoore[-_ ]?Threads\b|'
    r'\bIluvatar\b|\bBI[-_ ]?V?1\d{2}\b|\b天数智芯\b|'
    r'\bHygon\b|\bDCU\b|\b海光\b|\bMetaX\b|\bMuxi\b|\b沐曦\b|\bMXMACA\b|'
    # Chinese disclosure tokens that effectively imply a domestic-chip
    # disclosure even without a chip-name literal nearby. Limited to
    # phrases that only appear in genuine training-disclosure contexts
    # (TeleChat / Pangu / 国产化 sections).
    r'国产算力|国产框架|国产化算力|完全基于国产)',
    re.IGNORECASE,
)

HARDWARE_DURATION_RE = re.compile(
    r'(?:'
    # N × CHIP … training|compute|hours
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*(?:x\s*)?(?:A100|A800|A40|H100|H800|V100|H200|P100|T4|RTX\s*\d{4}|MI\d{3}[Xx]?|'
    r'TPU(?:\s*v\d+)?|GPU|Ascend(?:\s*\d{3}[A-Da-d]?)?|910[ABCDabcd]?|NPU|MLU\s*\d{3})\b.{0,60}?\b(?:training|compute|computation|hours?)|'
    # CHIP … training|compute|hours
    r'\b(?:A100|A800|A40|H100|H800|V100|H200|P100|T4|RTX\s*\d{4}|MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|Ascend(?:\s*\d{3}[A-Da-d]?)?|910[ABCDabcd]|MLU\s*\d{3})\b.{0,60}?'
    r'\b(?:training|compute|computation|hours?)|'
    # N gpu/npu hours … CHIP
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*(?:gpu|npu)\s+hours?\b.{0,80}?\b(?:A100|A800|A40|H100|H800|V100|H200|P100|T4|RTX\s*\d{4}|'
    r'MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|Ascend|910[ABCDabcd]|MLU\s*\d{3})|'
    # hours (of) … CHIP  — covers "1000 hours of A100"
    r'\b\d+(?:\.\d+)?\s*[MBK]?\s*hours?\b.{0,40}?\b(?:A100|A800|A40|H100|H800|V100|H200|P100|T4|RTX\s*\d{4}|'
    r'MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|GPU|Ascend|910[ABCDabcd]|NPU)|'
    # training|compute|computation (on|with|using) … CHIP  — covers "training on H100"
    r'\b(?:training|compute|computation|trained|fine[- ]?tuned|pre[- ]?trained)\b.{0,80}?'
    r'\b(?:A100|A800|A40|H100|H800|V100|H200|P100|T4|RTX\s*\d{4}|MI\d{3}[Xx]?|TPU(?:\s*v\d+)?|Gaudi|Trainium|'
    r'Ascend(?:\s*\d{3}[A-Da-d]?)?|910[ABCDabcd]|Atlas\s*\d{3}|MindSpore|MLU\s*\d{3}|Cambricon|'
    r'Kunlun(?:xin)?|P800|R[23]00|MUSA|MTT\s*S\d{3,4}|Moore[-_ ]?Threads|S4000|'
    r'Iluvatar|BI[-_ ]?V?1\d{2}|CoreX|Hygon|DCU|DTK|MetaX|Muxi|MXMACA|C5\d{2})\b'
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


# Distributed-training launcher patterns in train scripts / configs.
_TRAINING_LAUNCHER_RE = re.compile(
    r'(?:'
    # Shell launchers
    r'torchrun\s+[^\n]*--nproc[-_]?per[-_]?node|'
    r'python\s+-m\s+torch\.distributed|'
    r'accelerate\s+launch|'
    r'deepspeed\s+(?:--|\w+\.py)|'
    r'\bCUDA_VISIBLE_DEVICES\s*=\s*[\d,]+|'
    r'\bASCEND_RT_VISIBLE_DEVICES\s*=\s*[\d,]+|'
    r'--num[-_]?gpus?[=\s]+\d+|'
    r'--num[-_]?npus?[=\s]+\d+|'
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
    r'\bhccl\b|'                              # Huawei Collective Comm Lib (NCCL analog)
    r'--tensor[-_]?parallel|'
    r'--pipeline[-_]?parallel|'
    # Explicit CUDA training ops (distinct from runtime "device = cuda:0")
    r'\.cuda\(\)\s*$|'  # ".cuda()" at end of line is usually model .cuda() in training
    r'model\.cuda\(\)|'
    # TPU / XLA training
    r'import\s+jax|xm\.optimizer_step|xm\.xla_device|'
    r'torch_xla\.core|flax\.training|'
    # Ascend / MindSpore training
    r'import\s+mindspore|from\s+mindspore|'
    r'mindspore\.set_context|device_target\s*=\s*["\']?Ascend|'
    r'\.npu\(\)|model\.npu\(\)|'
    # Cambricon training
    r'import\s+torch_mlu|\.mlu\(\)|model\.mlu\(\)|'
    # Baidu Kunlun (XPU) training
    r'XPU_VISIBLE_DEVICES|paddle\.set_device\s*\(\s*["\']xpu|import\s+xpurt|'
    # Moore Threads (MUSA) training
    r'MUSA_VISIBLE_DEVICES|import\s+torch_musa|\.musa\(\)|model\.musa\(\)|'
    # Iluvatar (CoreX) training
    r'import\s+ixrt|import\s+corex|'
    # Hygon DCU training (typically rocm-ported but with hy-smi / DTK)
    r'DCU_VISIBLE_DEVICES|hy-smi|'
    # MetaX / Muxi training
    r'METAX_VISIBLE_DEVICES|import\s+mxmaca|mx-smi'
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
    # Order matters when a snippet hits multiple — the first match wins per snippet.
    # Put the non-CUDA stacks first so they override CUDA-mention noise.
    (re.compile(r'\b(?:jax\.distributed|TPUStrategy|torch_xla\.core|flax\.training|xm\.(?:optimizer_step|xla_device))\b', re.IGNORECASE), "google_tpu"),
    (re.compile(r'\b(?:rocm|hipify|rccl|MI\d{3}[Xx]?)\b', re.IGNORECASE), "amd"),
    (re.compile(
        r'\b(?:hccl|mindspore|ASCEND_RT_VISIBLE_DEVICES|davinci|npu-smi|torch_npu|'
        r'mindspore\.set_context|--num[-_]?npus?)\b|\.npu\(\)',
        re.IGNORECASE), "huawei_ascend"),
    (re.compile(r'\b(?:torch_mlu|cnnl|cndrv|bangpy|MLUDevice)\b|\.mlu\(\)', re.IGNORECASE), "cambricon"),
    (re.compile(
        r'\b(?:xpurt|kunlunxin|kunlun|XPU_VISIBLE_DEVICES)\b|'
        r'paddle\.set_device\s*\(\s*["\']xpu',
        re.IGNORECASE), "baidu_kunlun"),
    (re.compile(
        r'\b(?:torch_musa|mthreads|moore[-_ ]?threads|musatoolkit|musart|mccl|'
        r'MUSA_VISIBLE_DEVICES|mtgpu)\b|\.musa\(\)',
        re.IGNORECASE), "moore_threads"),
    (re.compile(
        r'\b(?:iluvatar|ixrt|corex|ixsmi)\b',
        re.IGNORECASE), "iluvatar"),
    (re.compile(
        r'\b(?:hygon|hy-smi|hygon[-_]dtk|DCU_VISIBLE_DEVICES)\b',
        re.IGNORECASE), "hygon"),
    (re.compile(
        r'\b(?:metax|muxi|mxmaca|mx-smi|METAX_VISIBLE_DEVICES)\b',
        re.IGNORECASE), "metax"),
    # Default: CUDA / PyTorch distributed / DeepSpeed / Accelerate → nvidia
    (re.compile(
        r'torchrun|CUDA_VISIBLE_DEVICES|torch\.distributed|DistributedDataParallel|'
        r'accelerate\s+launch|from\s+accelerate\s+import|\bAccelerator\s*\(|'
        r'deepspeed|\bnccl\b|init_process_group|ddp_find_unused_parameters|'
        r'--nproc[-_]?per[-_]?node|model\.cuda\(\)',
        re.IGNORECASE), "nvidia"),
]


def launcher_implied_chip(training_snippets):
    """Return the chip implied by a launcher pattern in a training-code file,
    or None. Non-CUDA stacks outrank CUDA since CUDA mentions leak into other
    vendors' repos as inference fallbacks."""
    if not training_snippets:
        return None
    hits = {"google_tpu": False, "amd": False, "huawei_ascend": False,
            "cambricon": False, "baidu_kunlun": False, "moore_threads": False,
            "iluvatar": False, "hygon": False, "metax": False, "nvidia": False}
    for s in training_snippets:
        src = s.get("source", "") or s.get("file", "") or s.get("section", "")
        if not _TRAINING_FILE_PATH_RE.search(src or ""):
            continue
        text = s.get("snippet", "") if isinstance(s, dict) else str(s)
        for rx, chip in _LAUNCHER_CHIP_CLASSIFIER:
            if rx.search(text):
                hits[chip] = True
                break
    for chip in ("google_tpu", "amd", "huawei_ascend", "cambricon",
                 "baidu_kunlun", "moore_threads", "iluvatar", "hygon", "metax",
                 "nvidia"):
        if hits[chip]:
            return chip
    return None


def extract_training_snippets(content, source="", max_snippets=4,
                              left=140, right=200):
    """Return [{"snippet", "source"}] for training-disclosure prose and
    training-launcher code patterns."""
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
