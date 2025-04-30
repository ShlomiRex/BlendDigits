import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import { FaGithub, FaEnvelope, FaExternalLinkAlt } from 'react-icons/fa';

function Model() {
    const GithubIcon = FaGithub as unknown as React.FC<{ size?: number }>;
    const EnvelopeIcon = FaEnvelope as unknown as React.FC<{ size?: number }>;
    const ExternalLinkAltIcon = FaExternalLinkAlt as unknown as React.FC<{ size?: number }>;

    const [output, setOutput] = useState<any>(null);
    const [inputImage1, setInputImage1] = useState<string>('');
    const [inputImage2, setInputImage2] = useState<string>('');
    const [session, setSession] = useState<ort.InferenceSession | null>(null);
    const [interpolation, setInterpolation] = useState<number>(0.5);

    // Function to get a random number between min and max (inclusive)
    const getRandomInt = (min: number, max: number) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    };

    // Function to load random MNIST images
    const loadRandomImages = () => {
        // Randomly select two different digits (1-9)
        let digit1 = getRandomInt(1, 9);
        let digit2 = getRandomInt(1, 9);
        
        // Ensure we have two different digits
        while (digit2 === digit1) {
            digit2 = getRandomInt(1, 9);
        }

        // Randomly select an image index (1-10) for each digit
        const index1 = getRandomInt(1, 10);
        const index2 = getRandomInt(1, 10);

        // Construct the image paths
        const newImage1 = `/mnist/${digit1}/${digit1}_${index1}.png`;
        const newImage2 = `/mnist/${digit2}/${digit2}_${index2}.png`;

        // Update the state with new images
        setInputImage1(newImage1);
        setInputImage2(newImage2);
    };

    // Load random images when component mounts
    useEffect(() => {
        loadRandomImages();
    }, []);

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
        <div className="model-container" style={{ 
            maxWidth: '1000px',
            margin: '0 auto',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '1.5rem',
            minHeight: '100vh',
            color: '#fff',
            width: '100%'
        }}>
            {/* Title */}
            <h1 style={{ 
                fontSize: '2.5rem',
                fontWeight: 'bold',
                textAlign: 'center',
                marginBottom: '0.25rem',
                background: 'linear-gradient(45deg, #00dbde, #fc00ff)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
            }}>
                BlendDigits
            </h1>

            {/* Description */}
            <p style={{ 
                fontSize: '1.1rem',
                color: '#a8b2d1',
                textAlign: 'center',
                maxWidth: '600px',
                marginBottom: '0.25rem'
            }}>
                Explore smooth transitions between handwritten digits — powered entirely in your browser.
            </p>

            {/* GitHub Link */}
            <a 
                href="https://github.com/ShlomiRex/BlendDigits" 
                target="_blank" 
                rel="noopener noreferrer"
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: '#64ffda',
                    textDecoration: 'none',
                    marginBottom: '1rem',
                    transition: 'color 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
                onMouseOut={(e) => e.currentTarget.style.color = '#64ffda'}
            >
                <GithubIcon size={20} />
                <span>View on GitHub</span>
            </a>

            {/* Images */}
            <div className="image-container" style={{ 
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '1.5rem',
                width: '100%',
                flexWrap: 'wrap',
                padding: '0.5rem'
            }}>
                <div className="image-item" style={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    <img 
                        src={inputImage1} 
                        alt="Input 1" 
                        style={{ 
                            width: '120px',
                            height: '120px',
                            objectFit: 'contain',
                            border: '2px solid #64ffda',
                            borderRadius: '8px',
                            background: '#0a192f',
                            padding: '0.5rem'
                        }} 
                    />
                    <span style={{ fontSize: '0.9rem', color: '#a8b2d1' }}>Input 1</span>
                </div>
                <div className="image-item" style={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    {outputImage ? (
                        <img 
                            src={outputImage} 
                            alt="Model Output" 
                            style={{ 
                                width: '120px',
                                height: '120px',
                                objectFit: 'contain',
                                border: '2px solid #64ffda',
                                borderRadius: '8px',
                                background: '#0a192f',
                                padding: '0.5rem'
                            }} 
                        />
                    ) : (
                        <div style={{ 
                            width: '120px',
                            height: '120px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            border: '2px solid #64ffda',
                            borderRadius: '8px',
                            background: '#0a192f',
                            color: '#a8b2d1'
                        }}>
                            <p>Loading...</p>
                        </div>
                    )}
                    <span style={{ fontSize: '0.9rem', color: '#a8b2d1' }}>Interpolation</span>
                </div>
                <div className="image-item" style={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    <img 
                        src={inputImage2} 
                        alt="Input 2" 
                        style={{ 
                            width: '120px',
                            height: '120px',
                            objectFit: 'contain',
                            border: '2px solid #64ffda',
                            borderRadius: '8px',
                            background: '#0a192f',
                            padding: '0.5rem'
                        }} 
                    />
                    <span style={{ fontSize: '0.9rem', color: '#a8b2d1' }}>Input 2</span>
                </div>
            </div>

            {/* Slider */}
            <div style={{ 
                width: '100%',
                maxWidth: '500px',
                padding: '0.5rem'
            }}>
                <div style={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                    width: '100%'
                }}>
                    <label style={{ 
                        fontSize: '1rem',
                        color: '#a8b2d1',
                        textAlign: 'center'
                    }}>
                        Interpolation: {interpolation.toFixed(2)}
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={interpolation}
                        onChange={handleSliderChange}
                        className="custom-slider"
                        style={{ 
                            width: '100%',
                            height: '8px',
                            borderRadius: '4px',
                            background: '#233554',
                            outline: 'none',
                            WebkitAppearance: 'none',
                            appearance: 'none',
                            border: '1px solid #64ffda'
                        }}
                    />
                </div>
            </div>

            {/* New Images Button */}
            <button
                onClick={loadRandomImages}
                style={{
                    padding: '0.5rem 1rem',
                    fontSize: '1rem',
                    backgroundColor: 'transparent',
                    color: '#64ffda',
                    border: '2px solid #64ffda',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(100, 255, 218, 0.1)'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
            >
                New images
            </button>

            {/* Description */}
            <div style={{
                maxWidth: '600px',
                padding: '1rem',
                textAlign: 'center',
                color: '#a8b2d1',
                fontSize: '0.95rem',
                lineHeight: '1.6',
                border: '1px solid rgba(100, 255, 218, 0.2)',
                borderRadius: '8px',
                backgroundColor: 'rgba(10, 25, 47, 0.3)',
                marginTop: '0.5rem'
            }}>
                BlendDigits is an interactive web demo that visualizes smooth transitions between two handwritten digits using a Variational Autoencoder (VAE). You select two MNIST images as input, and the model generates an interpolated digit in between. Everything runs locally in your browser — no server or cloud required.
            </div>

            {/* Warning Box */}
            <div style={{
                maxWidth: '600px',
                padding: '1rem',
                textAlign: 'center',
                color: '#ffd700',
                fontSize: '0.95rem',
                lineHeight: '1.6',
                border: '1px solid rgba(255, 215, 0, 0.3)',
                borderRadius: '8px',
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                marginTop: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
            }}>
                <span style={{ fontSize: '1.2rem' }}>⚠️</span>
                <span>On some phones, the browser doesn't support Web Assembly (WASM). In order for correct operation of the demo, please open the demo from a computer.</span>
            </div>

            {/* Footer Links */}
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.5rem',
                marginTop: '1rem'
            }}>
                <a 
                    href="mailto:shlomidom@gmail.com"
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        color: '#a8b2d1',
                        textDecoration: 'none',
                        transition: 'color 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.color = '#64ffda'}
                    onMouseOut={(e) => e.currentTarget.style.color = '#a8b2d1'}
                >
                    <span>Created by Shlomi Domnenco</span>
                    <EnvelopeIcon size={16} />
                </a>
                <a 
                    href="https://shlomidom.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        color: '#a8b2d1',
                        textDecoration: 'none',
                        transition: 'color 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.color = '#64ffda'}
                    onMouseOut={(e) => e.currentTarget.style.color = '#a8b2d1'}
                >
                    <span>Check out my professional portfolio</span>
                    <ExternalLinkAltIcon size={16} />
                </a>
            </div>
        </div>
    );
}

export default Model;
