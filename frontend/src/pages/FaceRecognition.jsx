import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/api';

/**
 * Face recognition attendance.
 *
 * The camera is the browser's own via navigator.mediaDevices.getUserMedia, so
 * it works the same on macOS, Windows and Linux and only needs a secure
 * context (https, or http://localhost). A captured frame is drawn to a canvas
 * and posted to the backend, which runs detection -> preprocessing ->
 * feature extraction -> scaler -> selector -> the trained model -> verification.
 */
export default function FaceRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const [cameraState, setCameraState] = useState('idle'); // idle | starting | live | error
  const [cameraError, setCameraError] = useState('');
  const [result, setResult] = useState(null);
  const [shot, setShot] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState('idle');
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const startCamera = async () => {
    setError('');
    setCameraError('');
    setCameraState('starting');
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraError('This browser does not support camera access.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState('live');
    } catch (err) {
      setCameraState('error');
      setCameraError(
        err?.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in your browser, then try again.'
          : err?.name === 'NotFoundError'
          ? 'No camera was found on this device.'
          : `Could not start the camera: ${err?.message || err}`
      );
    }
  };

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || cameraState !== 'live') return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, width, height);
    setShot(canvas.toDataURL('image/jpeg', 0.92));
    setResult(null);
    setError('');
  };

  const recognise = async (dataUrl) => {
    setBusy(true);
    setError('');
    try {
      const response = await api.predictBase64(dataUrl);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Recognition request failed.');
    } finally {
      setBusy(false);
    }
  };

  const handleFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      setShot(dataUrl);
      setResult(null);
      recognise(dataUrl);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  return (
    <div>
      <div className="page-head spread">
        <div>
          <h2>Face Recognition Attendance</h2>
          <p>Start the camera, position your face in the guide, then recognise.</p>
        </div>
        <button className="btn secondary" onClick={() => navigate('/attendance')}>
          View attendance
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="grid grid-2">
        <div className="card">
          <h3>Live camera</h3>

          <div className="camera-shell">
            <video ref={videoRef} playsInline muted style={{ display: cameraState === 'live' ? 'block' : 'none' }} />
            {cameraState === 'live' && (
              <div className="guide">
                <div className="guide-label">Position your face here</div>
              </div>
            )}
            {cameraState !== 'live' && (
              <div className="camera-placeholder">
                <div className="big">📷</div>
                {cameraState === 'starting' && 'Requesting camera permission…'}
                {cameraState === 'idle' && 'Camera is off. Click “Start camera” to begin.'}
                {cameraState === 'error' && <div className="alert error" style={{ marginTop: 14 }}>{cameraError}</div>}
              </div>
            )}
          </div>

          <div className="controls">
            {cameraState === 'live' ? (
              <>
                <button className="btn" onClick={capture}>📸 Capture frame</button>
                <button
                  className="btn success"
                  disabled={busy || !shot}
                  onClick={() => shot && recognise(shot)}
                >
                  {busy ? 'Recognising…' : '✅ Recognise'}
                </button>
                <button className="btn secondary" onClick={stopCamera}>⏹ Stop camera</button>
              </>
            ) : (
              <button className="btn" onClick={startCamera} disabled={cameraState === 'starting'}>
                ▶ Start camera
              </button>
            )}
            <button className="btn secondary" onClick={() => fileInputRef.current?.click()}>
              📁 Upload photo
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFile} />
          </div>

          <p className="small muted mt">
            The camera runs in your browser using <code>getUserMedia</code>. Frames are sent to the
            backend for recognition using the trained model; images are not stored.
          </p>
        </div>

        <div className="card">
          <h3>Result</h3>

          {result ? (
            <div className={`result ${result.status === 'recognized' ? 'recognized' : 'unknown'}`}>
              <div className="verdict">
                {result.status === 'recognized' ? '✓ Recognised' : '✕ Unknown student'}
              </div>
              {result.status === 'recognized' ? (
                <>
                  <div className="who">{result.student_name}</div>
                  <div className="meta">
                    ID <strong>{result.student_id}</strong>
                  </div>
                  <div className="meta">
                    {result.already_marked
                      ? 'Attendance already marked for today.'
                      : result.attendance_marked
                      ? 'Attendance marked automatically.'
                      : 'Attendance was not marked.'}
                  </div>
                </>
              ) : (
                <div className="meta">{result.message || 'Please try again.'}</div>
              )}

              <div className="mt" style={{ textAlign: 'left' }}>
                <div className="kv"><span className="k">Model confidence</span><span className="v">{result.confidence?.toFixed(4) ?? 'n/a'}</span></div>
                <div className="kv"><span className="k">Verified similarity</span><span className="v">{result.similarity?.toFixed(4) ?? 'n/a'}</span></div>
                <div className="kv"><span className="k">Threshold</span><span className="v">{result.threshold ?? 'n/a'}</span></div>
                <div className="kv"><span className="k">Model predicted</span><span className="v">{result.model_student_id ?? 'n/a'}</span></div>
                <div className="kv"><span className="k">Face detected</span><span className="v">{result.face_detected ? 'yes' : 'no'}</span></div>
                <div className="kv"><span className="k">Latency</span><span className="v">{result.latency_ms ? `${result.latency_ms} ms` : 'n/a'}</span></div>
              </div>

              {result.verification_reason && (
                <p className="small muted mt">{result.verification_reason}</p>
              )}
            </div>
          ) : (
            <div className="empty">
              <div style={{ fontSize: 40, marginBottom: 10 }}>🧑‍🎓</div>
              No recognition yet. Start the camera and capture a frame.
            </div>
          )}

          {shot && (
            <div className="mt">
              <h3>Captured frame</h3>
              <img className="shot" src={shot} alt="Captured frame" />
            </div>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} hidden />
    </div>
  );
}
