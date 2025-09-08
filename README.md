## embeddinggemma

Go bindings for Google’s EmbeddingGemma using ONNX Runtime.

🚧 WIP: This library is under active development. Expect breaking changes and incomplete coverage. Contributions and feedback are welcome.

### Features
- Minimal wrapper around ONNX Runtime for sentence embeddings
- Simple `Codec` built on SentencePiece for tokenization
- Example program under `cmd/` showing end‑to‑end usage

### Requirements
- Go 1.21+
- ONNX Runtime shared library installed locally
  - Set the environment variable `ONNX_RUNTIME` to the absolute path of the ONNX Runtime shared library for your platform (e.g., `libonnxruntime.dylib` on macOS)
- SentencePiece model file available for tokenization
  - Set `MODELPATH` to the absolute path of your `tokenizer.model` (a copy is included under `models/`)

### Install
```bash
go get github.com/gomithril/embeddinggemma@latest
```

### Models
This repository includes:
- `models/model.onnx` – EmbeddingGemma ONNX model
- `models/tokenizer.model` – SentencePiece tokenizer

You can replace these with compatible versions by pointing `MODELPATH` and the embedding `ModelPath` (see Config) to your desired files.

### Quick Start
Set the required environment variables and run the example:

```bash
# macOS example paths – adjust to your machine
export ONNX_RUNTIME=/usr/local/lib/libonnxruntime.dylib
export MODELPATH=$(pwd)/models/tokenizer.model

go run ./cmd
```

Expected output (truncated):
```
✅ Sentence embedding generated: length=768
🔍 First 10 dims: [ ... ]
```

### Usage
Programmatic usage mirrors the example in `cmd/main.go`:

```go
textCodec, err := codec.NewCodec()
if err != nil {
    log.Fatalf("failed to init codec: %v", err)
}

ids := textCodec.Encode("Mithril flows strong in Erebor")

svc := embedding.NewService(nil) // or pass a custom *embedding.Config
vec, err := svc.Generate(ids)
if err != nil {
    log.Fatalf("failed to generate embedding: %v", err)
}
fmt.Println(len(vec)) // embedding dimension
```

### Configuration
`embedding.DefaultConfig()` currently provides:
- `ModelPath`: `models/model.onnx`
- `SeqLen`: `16`
- `EmbedDim`: `768`

You can override by providing your own `*embedding.Config` to `embedding.NewService`.

Environment variables used:
- `ONNX_RUNTIME`: Absolute path to ONNX Runtime shared library
- `MODELPATH`: Absolute path to SentencePiece `tokenizer.model`

### Troubleshooting
- If you see an ONNX initialization error, verify `ONNX_RUNTIME` points to the correct shared library and that your system can load it (e.g., `otool -L` on macOS).
- If tokenization fails, ensure `MODELPATH` is set and the file exists.
- For runtime shape/type errors, confirm your `SeqLen` and `EmbedDim` match the model you are using.

### Roadmap (WIP)
- Batched inputs and configurable batch size
- Improved tokenizer utilities and normalization
- Optional pooling strategies for sentence embedding
- Model download helpers
- Benchmarks and accuracy checks

### License
Apache-2.0 (see LICENSE if present). If missing, treat as WIP and open an issue.
