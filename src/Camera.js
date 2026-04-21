import React, { useRef, useState, useEffect } from 'react';

export default function Camera() {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error(err);
        setError("Cannot access webcam. Please allow camera permission.");
      }
    }

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      {error && <p className="text-red-500">{error}</p>}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full max-w-md rounded-lg border"
      />
    </div>
  );
}