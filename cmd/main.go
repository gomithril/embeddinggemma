package main

import (
	"fmt"
	"os"

	"github.com/gomithril/embeddinggemma/codec"
	"github.com/gomithril/embeddinggemma/embedding"
	mylog "github.com/gomithril/embeddinggemma/internal/log"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Error().Err(err).Msg("Error loading .env file")
	}
	mylog.Init()

	data, err := os.ReadFile("input.txt")
	if err != nil {
		panic(fmt.Errorf("failed to read file: %w", err))
	}
	text := string(data)

	textCodec, err := codec.NewCodec()
	if err != nil {
		log.Error().Err(err).Msg("Failed to initialize text codec")
	}

	ids := textCodec.Encode(text)
	log.Info().Msg("Encoded Ids")

	svc, err := embedding.NewService(nil)
	if err != nil {
		log.Error().Err(err).Msg("unable to get new service")
	}
	defer svc.Close()
	chunkedIds := svc.ChunkText(ids)
	embeddings, err := svc.GenerateBatchConcurrently(chunkedIds, 8) // batchSize 8
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to generate embeddings")
	}
	log.Info().Msgf("Generated embeddings for %d chunks", len(embeddings))
}
