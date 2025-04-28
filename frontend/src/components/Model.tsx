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

    // Function to load and preprocess image
    const loadImageAsTensor = async (imageUrl: string): Promise<ort.Tensor> => {
        const image = new Image();
        image.src = imageUrl;

        // Wait for the image to load
        await new Promise((resolve) => {
            image.onload = resolve;
        });

        // Create a canvas to draw the image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get canvas context');
        }

        // Set the canvas size to match the image size (28x28 for MNIST)
        canvas.width = 28;
        canvas.height = 28;

        // Draw the image onto the canvas, scaling it to fit the 28x28 size
        ctx.drawImage(image, 0, 0, 28, 28);

        // Get the image data from the canvas
        const imageData = ctx.getImageData(0, 0, 28, 28);
        const data = imageData.data;

        // Normalize the image data (grayscale and normalize to [0, 1])
        const tensorData = new Float32Array(28 * 28);
        for (let i = 0; i < data.length; i += 4) {
            // Convert the pixel to grayscale (average of R, G, and B)
            const grayscale = (data[i] + data[i + 1] + data[i + 2]) / 3;
            tensorData[i / 4] = grayscale / 255; // Normalize to [0, 1]
        }

        // Return the image as an ONNX tensor
        return new ort.Tensor('float32', tensorData, [1, 1, 28, 28]);
    };

    // Function to run inference with two input tensors
    const runInference = async (tensor1: ort.Tensor, tensor2: ort.Tensor) => {
        if (session) {
            // Example load of latent vector from a JSON file
            const response = await fetch('/latent_vector.json');
            const latentArray = await response.json();
            const flatLatent = new Float32Array(latentArray.flat());
            const latentTensor = new ort.Tensor('float32', flatLatent, [1, latentArray[0].length]);

            console.log('Latent tensor shape:', latentTensor.dims);
            console.log('Latent tensor data:', latentTensor.data);

            console.log('Running inference...');

            const result = await session.run({ input: latentTensor });
            console.log('Inference completed');
            console.log('Result:', result);
            setOutput(result); // Store the result for rendering
        } else {
            console.error('Model session is not loaded yet');
        }
    };

    // Example of how to run inference with the given images
    useEffect(() => {
        if (session) {
            // Load and preprocess the images into tensors
            const loadAndRunInference = async () => {
                const tensor1 = await loadImageAsTensor(inputImage1);
                const tensor2 = await loadImageAsTensor(inputImage2);

                // Run inference with the loaded tensors
                runInference(tensor1, tensor2);
            };

            loadAndRunInference();
        }
    }, [session, inputImage1, inputImage2]);

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
