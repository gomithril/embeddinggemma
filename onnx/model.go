package onnx

import (
	ort "github.com/yalue/onnxruntime_go"
)

// ModelIO manages input and output tensors for ONNX models
type ModelIO struct {
	InputTensors  []ort.Value
	OutputTensors []ort.Value
}

// AddInput adds an input tensor to the model configuration
func (io *ModelIO) AddInput(tensor ort.Value) {
	io.InputTensors = append(io.InputTensors, tensor)
}

// AddOutput adds an output tensor to the model configuration
func (io *ModelIO) AddOutput(tensor ort.Value) {
	io.OutputTensors = append(io.OutputTensors, tensor)
}

func (io *ModelIO) Destroy() {
	for _, tensor := range io.InputTensors {
		tensor.Destroy()
	}

	for _, tensor := range io.OutputTensors {
		tensor.Destroy()
	}
}
