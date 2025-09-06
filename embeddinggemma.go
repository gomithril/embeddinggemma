package embeddinggemma

// Version of the library
const Version = "v0.0.1"

// Embedding is a placeholder for a vector
type Embedding []float32

// Embed returns a dummy embedding (to be implemented later).
func Embed(text string) (Embedding, error) {
    // TODO: implement actual model inference
    return Embedding{0.0, 0.1, 0.2}, nil
}
