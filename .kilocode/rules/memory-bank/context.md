# Context

## Current Focus
The project is currently in a beta state (v0.1.8). The main focus is on stability, testing, and potential performance optimizations (ONNX).

## Recent Changes
- Initial implementation of the Pipeline, EAST detector, and TRBA recognizer seems complete.
- Basic documentation and examples are in place.
- Setup configuration (`setup.py`) is ready for distribution.

## Next Steps
- **Testing:** Improve test coverage, specifically for the detector ("полное покрытие тестами детектора").
- **Robustness:** Verify handling of rotated boxes in the pipeline ("проверка повернутых боксов в пайплайне").
- **Optimization:** Investigate exporting models to ONNX for faster inference ("перевод в onnyx?").