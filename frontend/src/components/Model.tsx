import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

function Model() {
    const [output, setOutput] = useState<any>(null);
    const [inputImage1, setInputImage1] = useState<string>('/mnist/1/1_1.png');
    const [inputImage2, setInputImage2] = useState<string>('/mnist/2/2_1.png');
    const [session, setSession] = useState<ort.InferenceSession | null>(null);
    const [interpolation, setInterpolation] = useState<number>(0.5);

    // Function to load the ONNX model
    useEffect(() => {
        async function loadModel() {
            console.log('Loading ONNX model...');
            const modelSession = await ort.InferenceSession.create('model.onnx');
            console.log('Model loaded');
            setSession(modelSession);
        }

        loadModel();
    }, []);

    const loadImageAsTensor = async (imageUrl: string): Promise<ort.Tensor> => {
        const image = new Image();
        image.src = imageUrl;
        await new Promise((resolve) => {
            image.onload = resolve;
        });

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get canvas context');
        }

        canvas.width = 28;
        canvas.height = 28;
        ctx.drawImage(image, 0, 0, 28, 28);

        const imageData = ctx.getImageData(0, 0, 28, 28);
        const data = imageData.data;

        const tensorData = new Float32Array(28 * 28);
        for (let i = 0; i < data.length; i += 4) {
            const grayscale = (data[i] + data[i + 1] + data[i + 2]) / 3;
            tensorData[i / 4] = grayscale / 255;
        }

        return new ort.Tensor('float32', tensorData, [1, 1, 28, 28]);
    };

    const squeeze = (tensor: ort.Tensor, dim: number): ort.Tensor => {
        const newDims = tensor.dims.filter((_, idx) => idx !== dim);
        return new ort.Tensor(tensor.type, tensor.data, newDims);
    };

    // Updated runInference: no arguments!
    const runInference = async () => {
        if (!session) {
            console.error('Model session not loaded');
            return;
        }

        const tensor1 = await loadImageAsTensor(inputImage1);
        const tensor2 = await loadImageAsTensor(inputImage2);

        const squeezed1 = squeeze(tensor1, 0);
        const squeezed2 = squeeze(tensor2, 0);

        console.log("Running inference with interpolation =", interpolation);

        const input: Record<string, ort.Tensor> = {
            "input_img1": squeezed1,
            "input_img2": squeezed2,
            "interpolation": new ort.Tensor('float64', [interpolation]),
        };

        const result = await session.run(input);
        console.log('Inference completed');
        setOutput(result);
    };

    // Run once after session is ready
    useEffect(() => {
        if (session) {
            runInference();
        }
    }, [session, inputImage1, inputImage2]);

    // Function to convert output tensor to image data (base64)
    const convertTensorToImage = (outputTensor: any) => {
        const outputData = outputTensor.cpuData;
        const size = Math.sqrt(outputData.length);
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
            const value = outputData[i] * 255;
            imageData.data[i * 4] = value;
            imageData.data[i * 4 + 1] = value;
            imageData.data[i * 4 + 2] = value;
            imageData.data[i * 4 + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);

        return canvas.toDataURL();
    };

    const outputImage = output ? convertTensorToImage(output['output']) : null;

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        setInterpolation(value);
    };

    // Whenever interpolation changes, rerun inference
    useEffect(() => {
        if (session) {
            runInference();
        }
    }, [interpolation]);

    return (
        <div style={{ width: '100%' }}>
            {/* Images */}
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

            {/* Slider */}
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <label>Interpolation: {interpolation.toFixed(2)}</label>
                <br />
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={interpolation}
                    onChange={handleSliderChange}
                    style={{ width: '80%' }}
                />
            </div>
        </div>
    );
}

export default Model;
