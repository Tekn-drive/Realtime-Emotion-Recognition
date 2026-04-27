import './App.css';
import Camera from './Camera.js';
import { useState } from 'react';

function App() {
  const [rows, setRows] = useState([]);

  return (
    <div className="App">
      <header className="App-header">
        Realtime Emotion Recognition App 
      </header>
      <div className="table-and-camera"> 
          <div className="table-placeholder">
            <table className="emotion-table">
            <thead>
              <tr> 
                <th>Emotion</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
               {rows.map((row, index) => (
                <tr key={index}>
                  <td>{row.emotion}</td>
                  <td>{row.confidence.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
            </table>
          </div>
          <div className="camera-container">
            <Camera setRows={setRows} />
          </div>
        </div>
    </div>
  );
}

export default App;