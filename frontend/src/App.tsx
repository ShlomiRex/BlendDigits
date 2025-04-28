import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

import * as ort from 'onnxruntime-web';

function App() {
  const [output, setOutput] = useState<any>(null);
  const [inputImage1, setInputImage1] = useState<string>('path/to/your/input1.png');
  const [inputImage2, setInputImage2] = useState<string>('path/to/your/input2.png');
  
  useEffect(() => {
    async function loadModel() {
      const session = await ort.InferenceSession.create('model.onnx');
      const inputData = new Float32Array(28 * 28);
      const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
      const result = await session.run({ 'input': inputTensor });
      setOutput(result); // Store output to render the model's result
    }

    loadModel();
  }, []);

  // Assuming output is an image URL or image data that can be displayed as an img src
  const outputImage = output ? output['output'].data : null;

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
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
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
