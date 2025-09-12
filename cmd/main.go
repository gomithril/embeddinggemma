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

	svc := embedding.NewService(nil)
	chunkedIds := svc.ChunkText(ids)
	embeddings, err := svc.GenerateBatch(chunkedIds)
	if err != nil {
		log.Fatalln("Unable to generate embeddings")
	}
	fmt.Println(len(embeddings))
}
