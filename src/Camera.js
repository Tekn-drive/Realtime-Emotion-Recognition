import React, { useRef, useState, useEffect } from 'react';

export default function Camera() {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [box, setBox] = useState(null);

  useEffect(() => {
    let stream;
    let isProcessing = false;
    
    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        startSendingFrames();

      } catch (err) {
        console.error(err);
        setError("Cannot access webcam. Please allow camera permission.");
      }
    }

    async function startSendingFrames() {
      setInterval(async () => {
        if (isProcessing) return;

        isProcessing = true;

        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas) {
            isProcessing = false;
            return;
        }

        if (video.readyState !== 4) {
            isProcessing = false;
            return;
        }

        const ctx = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Capture current frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const base64Frame = canvas.toDataURL("image/jpeg", 0.7);

        const response = await fetch(
          "http://localhost:8000/predict_emotion",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              frame: base64Frame,
            }),
          }
        );

        const result = await response.json();

        // 🔥 Update UI from API response
        if (result.emotion) {
            setEmotion(result.emotion);
            setConfidence(result.confidence);
            setBox(result.box);
        }
        isProcessing = false;
      }, 100);
    }


    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

return (
    <div style={{ position: "relative"}}>
      {/* Webcam */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{
        width: "100%",
        height: "100%",
        objectFit: "cover"
        }}
      />

      {/* Hidden canvas for frame capture */}
      <canvas
        ref={canvasRef}
        style={{ display: "none" }}
      />

      {/* Bounding Box Overlay */}
      {box && (
        <div
          style={{
            position: "absolute",
            left: `${box.x}px`,
            top: `${box.y}px`,
            width: `${box.width}px`,
            height: `${box.height}px`,
            border: "2px solid red",
            pointerEvents: "none",
          }}
        >
          {/* Label */}
          <div
            style={{
              position: "absolute",
              top: "-30px",
              left: "0",
              background: "red",
              color: "white",
              padding: "4px 8px",
              fontSize: "14px",
              borderRadius: "4px",
            }}
          >
            {emotion} {confidence.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}