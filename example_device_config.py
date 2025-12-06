"""
Example: Using manuscript-ocr with different devices (CPU/CUDA/CoreML)
"""

from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA

# Option 1: CPU (default, works on all platforms)
print("1. CPU mode (default)")
pipeline_cpu = Pipeline()
print(f"   Detector providers: {pipeline_cpu.detector.runtime_providers()}")
print(f"   Recognizer providers: {pipeline_cpu.recognizer.runtime_providers()}")

# Option 2: NVIDIA CUDA (requires: pip install onnxruntime-gpu)
print("\n2. CUDA mode (NVIDIA GPU)")
try:
    detector_cuda = EAST(device="cuda")
    recognizer_cuda = TRBA(device="cuda")
    pipeline_cuda = Pipeline(detector=detector_cuda, recognizer=recognizer_cuda)
    print(f"   Detector providers: {detector_cuda.runtime_providers()}")
    print(f"   Recognizer providers: {recognizer_cuda.runtime_providers()}")
except Exception as e:
    print(f"   Error: {e}")
    print(f"   Install with: pip uninstall onnxruntime && pip install onnxruntime-gpu")

# Option 3: Apple Silicon CoreML (requires: pip install onnxruntime-silicon)
print("\n3. CoreML mode (Apple Silicon M1/M2/M3)")
try:
    detector_coreml = EAST(device="coreml")
    recognizer_coreml = TRBA(device="coreml")
    pipeline_coreml = Pipeline(detector=detector_coreml, recognizer=recognizer_coreml)
    print(f"   Detector providers: {detector_coreml.runtime_providers()}")
    print(f"   Recognizer providers: {recognizer_coreml.runtime_providers()}")
except Exception as e:
    print(f"   Error: {e}")
    print(
        f"   Install with: pip uninstall onnxruntime && pip install onnxruntime-silicon"
    )

print("\n✓ Device configuration examples completed")
