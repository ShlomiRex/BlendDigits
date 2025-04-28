import React, { useEffect, useRef, useState } from 'react';

function RandomCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Generate a random 28x28 grayscale image
        const imageData = ctx.createImageData(28, 28); // Create a blank 28x28 image
        for (let i = 0; i < imageData.data.length; i += 4) {
          // Each pixel in RGBA format (4 values per pixel)
          const randomValue = Math.random() * 255; // Random grayscale value
          imageData.data[i] = randomValue;      // Red
          imageData.data[i + 1] = randomValue;  // Green
          imageData.data[i + 2] = randomValue;  // Blue
          imageData.data[i + 3] = 255;         // Alpha (fully opaque)
        }
        ctx.putImageData(imageData, 0, 0); // Put the generated data on the canvas
        
        // Convert canvas data to image data URL
        const dataURL = canvas.toDataURL(); // Generate the data URL from the canvas
        setImageURL(dataURL); // Set the data URL to display the image
      }
    }
  }, []);

  return (
    <div>
      <h1>Random 28x28 Image from Canvas (Scaled)</h1>
      <canvas
        ref={canvasRef}
        width={28} // Set the internal canvas size to 28x28
        height={28} // Set the internal canvas size to 28x28
        style={{ width: '280px', height: '280px' }} // Scale the canvas visually by 10x
      ></canvas>

      {imageURL && (
        <div>
          <h2>Generated Image</h2>
          <img src={imageURL} 
            alt="Generated Random Image" 
            style={{ width: '280px', height: '280px', imageRendering: 'pixelated' }} />
        </div>
      )}
    </div>
  );
}

export default RandomCanvas;
