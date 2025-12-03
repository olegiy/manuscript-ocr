from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
import shutil
import urllib.request
import tempfile
import onnxruntime as ort

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class BaseModel(ABC):
    """
    Base class for models with unified interface:
        - artifact loading (local, URL, GitHub, GDrive, preset)
        - device selection
        - backend session initialization
        - inference / training / export
    """

    default_weights_name: Optional[str] = None
    pretrained_registry: Dict[str, str] = {}

    def __init__(self, weights: Optional[str] = None, device: Optional[str] = None, **kwargs):
        self.device = self._resolve_device(device)
        self.weights = self._resolve_weights(weights)
        self.extra_config = kwargs
        self.session = None

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------
    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device

        try:
            if ort.get_device().upper() == "GPU":
                return "cuda"
        except Exception:
            pass

        return "cpu"

    def runtime_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]

    # -------------------------------------------------------------------------
    # WEIGHT RESOLUTION (main artifact)
    # -------------------------------------------------------------------------
    def _resolve_weights(self, weights: Optional[str]) -> str:
        if weights is None:
            if not self.default_weights_name:
                raise ValueError(f"{self.__class__.__name__} must define default_weights_name")
            weights = self.default_weights_name

        w = str(weights)

        # 1. Local file
        if Path(w).expanduser().exists():
            return str(Path(w).expanduser().absolute())

        # 2. URL
        if w.startswith(("http://", "https://")):
            return self._download_http(w)

        # 3. GitHub
        if w.startswith("github://"):
            return self._download_github(w)

        # 4. Google Drive
        if w.startswith("gdrive:"):
            return self._download_gdrive(w)

        # 5. Preset registry
        if w in self.pretrained_registry:
            return self._resolve_weights(self.pretrained_registry[w])

        raise ValueError(
            f"Unknown weights '{weights}'. Supported: local path, URL, "
            f"github://.., gdrive:ID, presets={list(self.pretrained_registry.keys())}"
        )

    # -------------------------------------------------------------------------
    # GENERIC EXTRA ARTIFACT RESOLUTION
    # -------------------------------------------------------------------------
    def _resolve_extra_artifact(
        self,
        value: Optional[str],
        *,
        default_name: Optional[str],
        registry: Dict[str, str],
        description: str = "artifact",
    ) -> str:
        """
        Universal resolver for auxiliary artifacts (config, charset, vocab, etc.).

        Supports:
            - None → use default_name
            - local path
            - URL
            - github://
            - gdrive:
            - preset (lookup in registry)
        """

        # 0) Default
        if value is None:
            if default_name is None:
                raise ValueError(f"{self.__class__.__name__}: no default {description} defined.")
            value = default_name

        v = str(value)

        # 1) Local file
        if Path(v).expanduser().exists():
            return str(Path(v).expanduser().absolute())

        # 2) URL
        if v.startswith(("http://", "https://")):
            return self._download_http(v)

        # 3) GitHub
        if v.startswith("github://"):
            return self._download_github(v)

        # 4) Google Drive
        if v.startswith("gdrive:"):
            return self._download_gdrive(v)

        # 5) Preset
        if v in registry:
            return self._resolve_extra_artifact(
                registry[v],
                default_name=default_name,
                registry=registry,
                description=description
            )

        raise ValueError(
            f"Unknown {description} '{value}'. "
            f"Supported: local file, URL, github://, gdrive:, presets={list(registry.keys())}"
        )

    # -------------------------------------------------------------------------
    # DOWNLOAD HELPERS
    # -------------------------------------------------------------------------
    @property
    def _cache_dir(self) -> Path:
        d = Path.home() / ".manuscript" / "weights"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _download_http(self, url: str) -> str:
        """Download file from HTTP/HTTPS URL"""
        file = self._cache_dir / Path(url).name
        if file.exists():
            return str(file)

        print(f"Downloading {Path(url).name} from {url}")
        
        # Create temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False).name
        
        if tqdm is not None:
            try:
                # Get file size
                with urllib.request.urlopen(url) as response:
                    total_size = int(response.headers.get('content-length', 0))
                
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                         desc=Path(url).name, ncols=80) as pbar:
                    def reporthook(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if downloaded < total_size:
                            pbar.update(block_size)
                        else:
                            pbar.update(total_size - pbar.n)
                    
                    urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
            except Exception as e:
                # Fallback to simple download if progress bar fails
                print(f"Progress bar error ({e}), downloading without progress...")
                urllib.request.urlretrieve(url, tmp)
        else:
            # No tqdm available, simple download
            urllib.request.urlretrieve(url, tmp)
        
        # Move to cache
        shutil.move(tmp, file)
        print(f"✓ Downloaded to {file}")
        return str(file)

    def _download_github(self, spec: str) -> str:
        payload = spec.replace("github://", "").strip()
        owner, repo, tag, *path_parts = payload.split("/")
        url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{'/'.join(path_parts)}"
        return self._download_http(url)

    def _download_gdrive(self, spec: str) -> str:
        """Download file from Google Drive with progress bar."""
        file_id = spec.split("gdrive:", 1)[1]
        
        # Check if gdown is available for better GDrive support
        try:
            import gdown
            
            # Extract filename from cache or use file_id
            file = self._cache_dir / f"{file_id}.bin"  # Will be renamed after download
            
            print(f"Downloading from Google Drive (ID: {file_id})")
            output = gdown.download(id=file_id, output=str(file), quiet=False)
            
            if output is None:
                raise RuntimeError(f"Failed to download from Google Drive: {file_id}")
            
            return output
        except ImportError:
            # Fallback to direct URL (may not work for large files)
            print("Warning: gdown not installed. Using direct URL (may fail for large files)")
            print("Install gdown for better Google Drive support: pip install gdown")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return self._download_http(url)

    # -------------------------------------------------------------------------
    # BACKEND INITIALIZATION
    # -------------------------------------------------------------------------
    @abstractmethod
    def _initialize_session(self):
        ...

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    @abstractmethod
    def predict(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    # -------------------------------------------------------------------------
    # OPTIONAL API
    # -------------------------------------------------------------------------
    @staticmethod
    def train(*args, **kwargs):
        raise NotImplementedError("This model does not support training.")

    @staticmethod
    def export(*args, **kwargs):
        raise NotImplementedError("This model does not support export.")
