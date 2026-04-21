import './App.css';
import Camera from './Camera.js';

function App() {
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
              <tr>
                <td>Happy</td>
                <td>0.85</td>
              </tr>
              <tr>
                <td>Sad</td>
                <td>0.10</td>
              </tr>
              <tr>
                <td>Neutral</td>
                <td>0.05</td>
              </tr>
            </tbody>
            </table>
          </div>

          <div className="camera-container">
            <Camera />
          </div>
        </div>
    </div>
  );
}

export default App;