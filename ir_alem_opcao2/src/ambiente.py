"""Detecção do ambiente e escolha do backend do Keras.

Este módulo NÃO importa o keras: o backend do Keras 3 só pode ser definido antes do primeiro
`import keras`, e não pode ser trocado depois (https://keras.io/getting_started/).

Por que existe: no runtime "latest" do Colab (out/2026, CUDA 13) o TensorFlow pré-instalado
provavelmente não enxerga a T4, enquanto o PyTorch enxerga. O código dos modelos usa só a API
do Keras 3, então roda igual nos backends tensorflow, torch e jax; aqui escolhemos o melhor
disponível e registramos a escolha.

Com o backend torch, classificadores e segmentador (ultralytics) usam o mesmo framework no mesmo processo.
"""
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path


def no_colab() -> bool:
    """True quando o notebook roda no Google Colab."""
    return "google.colab" in sys.modules


def _tem_modulo(nome: str) -> bool:
    return importlib.util.find_spec(nome) is not None


def tensorflow_tem_gpu() -> bool:
    """Testa, em um subprocesso, se o TensorFlow instalado enxerga alguma GPU.

    Subprocesso porque importar o TF aqui ocuparia a GPU e fixaria o estado do processo.
    """
    if not _tem_modulo("tensorflow"):
        return False
    codigo = "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
    try:
        saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=300)
        return saida.stdout.strip().splitlines()[-1] != "0"
    except Exception:                                            # TF quebrado conta como "sem GPU"
        return False


def torch_tem_gpu() -> bool:
    if not _tem_modulo("torch"):
        return False
    import torch
    return torch.cuda.is_available()


def preparar_backend(preferido: str | None = None) -> str:
    """Define KERAS_BACKEND antes do import do keras e devolve o backend escolhido.

    Ordem de preferência:
        1. argumento `preferido` ou variável FARMTECH_BACKEND;
        2. 'tensorflow' se o TF enxerga GPU (ex.: runtime fixo 2026.07 do Colab);
        3. 'torch' se instalado: usa a GPU no runtime latest do Colab e foi o mais rápido em CPU
           num teste local (~9 s contra ~32 s por época do JAX, mesma CNN, versão inicial do código);
        4. 'jax', se instalado;
        5. 'tensorflow' (CPU) como último recurso.
    """
    escolha = preferido or os.environ.get("FARMTECH_BACKEND")
    if not escolha:
        if tensorflow_tem_gpu():
            escolha = "tensorflow"
        elif _tem_modulo("torch"):
            escolha = "torch"
        elif _tem_modulo("jax"):
            escolha = "jax"
        else:
            escolha = "tensorflow"
    if "keras" in sys.modules:
        import keras
        if keras.config.backend() != escolha:
            raise RuntimeError(
                f"O keras já foi importado com backend '{keras.config.backend()}'. "
                "Reinicie o runtime e execute esta célula antes de qualquer import do keras.")
    os.environ["KERAS_BACKEND"] = escolha
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")     # TF não reserva toda a VRAM
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # idem para o JAX
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")               # silencia logs C++ do TF
    return escolha


def commit_git() -> str:
    """Commit do repositório que gerou os resultados (rastreabilidade)."""
    raiz = Path(__file__).resolve().parents[2]
    try:
        saida = subprocess.run(["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=30)
        sujo = subprocess.run(["git", "-C", str(raiz), "status", "--porcelain", "--", "ir_alem_opcao2/src"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        return saida.stdout.strip() + (" (src/ com alterações não commitadas)" if sujo else "")
    except Exception:
        return "desconhecido"


def descrever_ambiente() -> dict:
    """Versões e dispositivos, para registrar junto com os resultados.

    As versões são lidas dos metadados dos pacotes, sem importá-los: importar o keras sem backend
    definido (ou o TensorFlow junto do PyTorch) teria efeitos colaterais no processo.
    """
    info = {"python": platform.python_version(), "colab": no_colab(),
            "keras_backend": os.environ.get("KERAS_BACKEND"), "commit": commit_git()}
    for pacote in ("keras", "tensorflow", "torch", "jax", "ultralytics", "numpy", "pandas", "scikit-learn"):
        try:
            info[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            pass
    info["gpu_torch"] = torch_tem_gpu()
    if info["gpu_torch"]:
        import torch
        info["gpu_nome"] = torch.cuda.get_device_name(0)
    return info


def exportar(arquivos: list[str], info: dict, nome_json: str) -> None:
    """Copia os CSVs da execução para ir_alem_opcao2/resultados/ e registra o ambiente.

    No Colab, resultados/ e assets/ ficam no clone temporário em /content: eles também são copiados
    para o Drive (DADOS_DIR/para_o_repo), de onde podem ser trazidos para o repositório.
    """
    import json
    import shutil
    from . import config
    for nome in arquivos:
        shutil.copy(config.RESULTADOS_DIR / nome, config.RESULTADOS_REPO / nome)
    texto = json.dumps(info, indent=1, ensure_ascii=False, default=str)
    for pasta in (config.RESULTADOS_REPO, config.RESULTADOS_DIR):  # o notebook 03 confere o do 01
        (pasta / nome_json).write_text(texto, encoding="utf-8")
    print("✅ Resultados copiados para", config.RESULTADOS_REPO)
    if no_colab():
        destino = config.DADOS_DIR / "para_o_repo"
        shutil.copytree(config.RESULTADOS_REPO, destino / "resultados", dirs_exist_ok=True)
        shutil.copytree(config.ASSETS_DIR, destino / "assets", dirs_exist_ok=True)
        print("📥 Cópia de resultados/ e assets/ salva no Drive em", destino)


def dispositivo_keras() -> str:
    """Rótulo curto do dispositivo efetivamente usado pelo Keras (para tabelas de latência)."""
    import keras
    backend = keras.config.backend()
    if backend == "torch":
        import torch
        return f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    if backend == "tensorflow":
        import tensorflow as tf
        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    if backend == "jax":
        import jax
        return jax.devices()[0].platform.upper()
    return backend
