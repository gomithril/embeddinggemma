package codec

import (
	"fmt"
	"os"
	"sync"

	"github.com/eliben/go-sentencepiece"
)

var (
	once    sync.Once
	proc    *sentencepiece.Processor
	loadErr error
)

func NewProcessor(modelPath string) (*sentencepiece.Processor, error) {
	once.Do(func() {
		if modelPath == "" {
			loadErr = fmt.Errorf("model path is empty")
			return
		}

		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			loadErr = fmt.Errorf("model file not found at %s", modelPath)
			return
		}

		proc, loadErr = sentencepiece.NewProcessorFromPath(modelPath)
	})
	if loadErr != nil {
		return nil, loadErr
	}

	return proc, nil
}

type Codec struct {
	processor *sentencepiece.Processor
}

func NewCodec() (*Codec, error) {
	protoFile := os.Getenv("MODELPATH")
	proc, err := NewProcessor(protoFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load SentencePiece processor: %v", err)
	}

	return &Codec{processor: proc}, nil
}

func (c *Codec) Encode(text string) []int64 {
	tokens := c.processor.Encode(text)

	ids := make([]int64, len(tokens))
	for i, token := range tokens {
		ids[i] = int64(token.ID)
	}
	return ids
}
