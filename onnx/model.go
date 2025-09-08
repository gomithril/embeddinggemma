package onnx

import (
	ort "github.com/yalue/onnxruntime_go"
)

// ModelIO manages input and output tensors for ONNX models
type ModelIO struct {
	InputNames    []string
	OutputNames   []string
	InputTensors  []ort.Value
	OutputTensors []ort.Value
}

// AddInput adds an input tensor to the model configuration
func (io *ModelIO) AddInput(name string, tensor ort.Value) {
	io.InputNames = append(io.InputNames, name)
	io.InputTensors = append(io.InputTensors, tensor)
}

// AddOutput adds an output tensor to the model configuration
func (io *ModelIO) AddOutput(name string, tensor ort.Value) {
	io.OutputNames = append(io.OutputNames, name)
	io.OutputTensors = append(io.OutputTensors, tensor)
}

// AttachToSession creates a new session with the configured inputs and outputs
func (io *ModelIO) AttachToSession(modelPath string) (*ort.AdvancedSession, error) {
	return ort.NewAdvancedSession(
		modelPath,
		io.InputNames,
		io.OutputNames,
		io.InputTensors,
		io.OutputTensors,
		nil,
	)
}
