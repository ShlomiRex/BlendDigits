import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

function Model() {
    const [output, setOutput] = useState<any>(null);
    const [inputImage1, setInputImage1] = useState<string>('/mnist/1/1_1.png');
    const [inputImage2, setInputImage2] = useState<string>('/mnist/2/2_1.png');
    const [session, setSession] = useState<ort.InferenceSession | null>(null);

    // Function to load the ONNX model
    useEffect(() => {
        async function loadModel() {
            console.log('Loading ONNX model...');
            const modelSession = await ort.InferenceSession.create('model.onnx');
            console.log('Model loaded');
            setSession(modelSession); // Store the session for later inference
        }

        loadModel();
    }, []);

    // Function to run inference with two input tensors
    const runInference = async (tensor1: ort.Tensor, tensor2: ort.Tensor) => {
        if (session) {
            console.log('Running inference...');
            const result = await session.run({ input1: tensor1, input2: tensor2 });
            console.log('Inference completed');
            console.log('Result:', result);
            setOutput(result); // Store the result for rendering
        } else {
            console.error('Model session is not loaded yet');
        }
    };

    // // Example of how to run inference with random tensors
    // useEffect(() => {
    //     if (session) {
    //         // Generate example input tensors
    //         const inputData1 = new Float32Array(28 * 28);
    //         const inputTensor1 = new ort.Tensor('float32', inputData1, [1, 1, 28, 28]);

    //         const inputData2 = new Float32Array(28 * 28);
    //         const inputTensor2 = new ort.Tensor('float32', inputData2, [1, 1, 28, 28]);

    //         // Run inference with the generated tensors
    //         runInference(inputTensor1, inputTensor2);
    //     }
    // }, [session]);

    // Function to convert output tensor to image data (base64)
    const convertTensorToImage = (outputTensor: any) => {
        console.log('Converting tensor to image...');
        const outputData = outputTensor.cpuData;
        const size = Math.sqrt(outputData.length); // In this case, 28x28
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            console.error('Failed to get canvas context');
            return null;
        }

        const imageData = ctx.createImageData(size, size);

        for (let i = 0; i < outputData.length; i++) {
            const value = outputData[i] * 255; // Convert to range [0, 255]
            imageData.data[i * 4] = value; // Red
            imageData.data[i * 4 + 1] = value; // Green
            imageData.data[i * 4 + 2] = value; // Blue
            imageData.data[i * 4 + 3] = 255; // Alpha
        }

        ctx.putImageData(imageData, 0, 0);

        return canvas.toDataURL();
    };

    const outputImage = output ? convertTensorToImage(output['output']) : null;

    return (
        <div className="image-container" style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <div className="image-item" style={{ margin: '10px' }}>
                <img src={inputImage1} alt="Input 1" style={{ width: '100px', height: '100px' }} />
            </div>
            <div className="image-item" style={{ margin: '10px' }}>
                {outputImage ? (
                    <img src={outputImage} alt="Model Output" style={{ width: '100px', height: '100px' }} />
                ) : (
                    <p>Loading...</p>
                )}
            </div>
            <div className="image-item" style={{ margin: '10px' }}>
                <img src={inputImage2} alt="Input 2" style={{ width: '100px', height: '100px' }} />
            </div>
        </div>
    );
}

export default Model;
