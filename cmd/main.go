package main

import (
	"fmt"
	"log"

	"github.com/gomithril/embeddinggemma/codec"
	"github.com/gomithril/embeddinggemma/embedding"
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}

	textCodec, err := codec.NewCodec()
	if err != nil {
		log.Fatalf("Failed to initialize text codec: %v", err)
	}

	text := "Mithril flows strong in Erebor"

	ids := textCodec.Encode(text)

	// Create embedding service with default config
	service := embedding.NewService(nil)

	// Generate embeddings
	embeddings, err := service.Generate(ids)
	if err != nil {
		log.Fatalf("Failed to generate embeddings: %v", err)
	}

	fmt.Printf("✅ Sentence embedding generated: length=%d\n", len(embeddings))
	if len(embeddings) >= 10 {
		fmt.Printf("🔍 First 10 dims: %v\n", embeddings[:10])
	}
}
