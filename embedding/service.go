package embedding

import (
	"fmt"
	"os"

	"github.com/gomithril/embeddinggemma/onnx"
	ort "github.com/yalue/onnxruntime_go"
)

// Config holds embedding service configuration
type Config struct {
	ModelPath string
	SeqLen    int64
	EmbedDim  int64
}

// DefaultConfig returns default embedding configuration
func DefaultConfig() *Config {
	return &Config{
		ModelPath: "models/model.onnx",
		SeqLen:    512,
		EmbedDim:  768,
	}
}

// Service handles embedding operations
type Service struct {
	config *Config
}

// NewService creates a new embedding service
func NewService(config *Config) *Service {
	if config == nil {
		config = DefaultConfig()
	}
	return &Service{config: config}
}

// initializeONNXRuntime sets up the ONNX runtime environment
func (s *Service) initializeONNXRuntime() error {
	// Initialize ONNX runtime
	ONNX_RUNTIME := os.Getenv("ONNX_RUNTIME")
	ort.SetSharedLibraryPath(ONNX_RUNTIME)
	err := ort.InitializeEnvironment()
	return err
}

// prepareTensors creates and prepares input/output tensors for inference
func (s *Service) prepareTensors(ids []int64) (*onnx.ModelIO, error) {
	seqLen := s.config.SeqLen
	batchSize := int64(1)

	// Prepare padded IDs and attention mask
	paddedIds := make([]int64, seqLen)
	attMask := make([]int64, seqLen)

	for i := 0; i < int(seqLen) && i < len(ids); i++ {
		paddedIds[i] = ids[i]
		attMask[i] = 1 // mark actual tokens
	}

	// Create input tensors
	inputIdsTensor, err := ort.NewTensor(ort.NewShape(batchSize, seqLen), paddedIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %v", err)
	}

	attMaskTensor, err := ort.NewTensor(ort.NewShape(batchSize, seqLen), attMask)
	if err != nil {
		inputIdsTensor.Destroy()
		return nil, fmt.Errorf("failed to create attention_mask tensor: %v", err)
	}

	// Create output tensors
	tokenEmbedsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batchSize, seqLen, s.config.EmbedDim))
	if err != nil {
		inputIdsTensor.Destroy()
		attMaskTensor.Destroy()
		return nil, fmt.Errorf("failed to create token_embeddings tensor: %v", err)
	}

	sentenceEmbedTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batchSize, s.config.EmbedDim))
	if err != nil {
		inputIdsTensor.Destroy()
		attMaskTensor.Destroy()
		tokenEmbedsTensor.Destroy()
		return nil, fmt.Errorf("failed to create sentence_embedding tensor: %v", err)
	}

	// Create ModelIO and configure inputs/outputs
	io := &onnx.ModelIO{}
	io.AddInput("input_ids", inputIdsTensor)
	io.AddInput("attention_mask", attMaskTensor)
	io.AddOutput("token_embeddings", tokenEmbedsTensor)
	io.AddOutput("sentence_embedding", sentenceEmbedTensor)

	return io, nil
}

// Generate creates embeddings for the given token IDs
func (s *Service) Generate(ids []int64) ([]float32, error) {
	// Initialize ONNX runtime
	if err := s.initializeONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to init ONNX runtime environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Prepare tensors
	io, err := s.prepareTensors(ids)
	if err != nil {
		return nil, err
	}

	// Ensure cleanup of tensors
	defer func() {
		for _, tensor := range io.InputTensors {
			tensor.Destroy()
		}
		for _, tensor := range io.OutputTensors {
			tensor.Destroy()
		}
	}()

	// Create session
	session, err := io.AttachToSession(s.config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}
	defer session.Destroy()

	// Run inference
	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	sentenceEmbedValue := io.OutputTensors[len(io.OutputTensors)-1]
	sentenceEmbedTensor, ok := sentenceEmbedValue.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("failed to type assert output tensor to *ort.Tensor[float32]")
	}
	sentenceEmbedding := sentenceEmbedTensor.GetData()

	return sentenceEmbedding, nil
}

func (s *Service) prepareBatchTensors(batch [][]int64) (*onnx.ModelIO, error) {
	batchSize := int64(len(batch))
	seqLen := s.config.SeqLen

	// Prepare padded IDs and masks
	paddedIds := make([]int64, batchSize*seqLen)
	attMask := make([]int64, batchSize*seqLen)

	for b, ids := range batch {
		for i := 0; i < int(seqLen) && i < len(ids); i++ {
			paddedIds[b*int(seqLen)+i] = ids[i]
			attMask[b*int(seqLen)+i] = 1
		}
	}

	// Create input tensors
	inputIdsTensor, err := ort.NewTensor(ort.NewShape(batchSize, seqLen), paddedIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %v", err)
	}

	attMaskTensor, err := ort.NewTensor(ort.NewShape(batchSize, seqLen), attMask)
	if err != nil {
		inputIdsTensor.Destroy()
		return nil, fmt.Errorf("failed to create attention_mask tensor: %v", err)
	}

	// Create output tensors
	tokenEmbedsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batchSize, seqLen, s.config.EmbedDim))
	if err != nil {
		inputIdsTensor.Destroy()
		attMaskTensor.Destroy()
		return nil, fmt.Errorf("failed to create token_embeddings tensor: %v", err)
	}

	sentenceEmbedTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batchSize, s.config.EmbedDim))
	if err != nil {
		inputIdsTensor.Destroy()
		attMaskTensor.Destroy()
		tokenEmbedsTensor.Destroy()
		return nil, fmt.Errorf("failed to create sentence_embedding tensor: %v", err)
	}

	io := &onnx.ModelIO{}
	io.AddInput("input_ids", inputIdsTensor)
	io.AddInput("attention_mask", attMaskTensor)
	io.AddOutput("token_embeddings", tokenEmbedsTensor)
	io.AddOutput("sentence_embedding", sentenceEmbedTensor)

	return io, nil
}

// GenerateBatch processes multiple sequences at once
func (s *Service) GenerateBatch(batch [][]int64) ([][]float32, error) {
	if err := s.initializeONNXRuntime(); err != nil {
		return nil, fmt.Errorf("failed to init ONNX runtime: %v", err)
	}

	io, err := s.prepareBatchTensors(batch)
	if err != nil {
		return nil, err
	}

	defer func() {
		for _, tensor := range io.InputTensors {
			tensor.Destroy()
		}

		for _, tensor := range io.OutputTensors {
			tensor.Destroy()
		}
	}()

	session, err := io.AttachToSession(s.config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	// Extract sentence embeddings for each batch element
	sentenceEmbedValue := io.OutputTensors[len(io.OutputTensors)-1]
	sentenceEmbedTensor, ok := sentenceEmbedValue.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("failed to type assert sentence embedding tensor")
	}
	allEmbeds := sentenceEmbedTensor.GetData()

	batchSize := len(batch)
	embedDim := int(s.config.EmbedDim)
	results := make([][]float32, batchSize)

	for b := 0; b < batchSize; b++ {
		start := b * embedDim
		end := start + embedDim
		results[b] = allEmbeds[start:end]
	}

	return results, nil
}

func (s *Service) ChunkText(ids []int64) [][]int64 {
	var chunks [][]int64
	for i:=0; i<len(ids); i += int(s.config.SeqLen) {
		end := min(i + int(s.config.SeqLen), len(ids))
		chunks = append(chunks, ids[i:end])
	}
	return chunks
}
