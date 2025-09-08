package onnx

import (
	ort "github.com/yalue/onnxruntime_go"
)

// NewAdvancedSession creates a new ONNX runtime advanced session
func NewAdvancedSession(
	modelPath string,
	inputs []string,
	outputs []string,
	inputTensors []ort.Value,
	outputTensors []ort.Value,
) (*ort.AdvancedSession, error) {
	return ort.NewAdvancedSession(
		modelPath,
		inputs,
		outputs,
		inputTensors,
		outputTensors,
		nil,
	)
}
