package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gomithril/embeddinggemma/codec"
	"github.com/gomithril/embeddinggemma/embedding"
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}

	data, err := os.ReadFile("input.txt")
	if err != nil {
		panic(fmt.Errorf("failed to read file: %w", err))
	}
	text := string(data)

	textCodec, err := codec.NewCodec()
	if err != nil {
		log.Fatalf("Failed to initialize text codec: %v", err)
	}

	ids := textCodec.Encode(text)

	svc, err := embedding.NewService(nil)
	if err != nil {
		log.Fatalf("unable to get new service: %v", err)
	}
	defer svc.Close()
	chunkedIds := svc.ChunkText(ids)
	embeddings, err := svc.GenerateBatch(chunkedIds)
	if err != nil {
		log.Fatalf("Unable to generate embeddings: %v", err)
	}
	fmt.Println(len(embeddings))
}
