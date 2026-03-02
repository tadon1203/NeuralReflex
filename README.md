# NeuralReflex

NeuralReflex is an **AI aim-assist software project targeting low latency**.  
It is building a foundation for stable inference and overlay execution with GPU resource sharing and lightweight runtime loops.

## Technology Stack
- In use:
  - DirectX 11
  - DirectX 12
  - spdlog
- Planned:
  - ONNX Runtime + DirectML Execution Provider (DML EP)
  - ImGui

## Directory Layout
- `src/core`: application lifecycle control (startup, loop, shutdown)
- `src/gfx`: DirectX context management
- `src/utils`: logging and error-handling utilities
- `include/nrx/*`: public headers
- `build/`: build outputs

## Build
```bash
cmake --preset debug
cmake --build --preset debug
```

With `CMAKE_EXPORT_COMPILE_COMMANDS=ON`, `compile_commands.json` is copied to `build/compile_commands.json`.
