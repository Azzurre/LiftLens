import { useState } from "react";
import "./App.css";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const normaliseAnalysisResponse = (data) => {
    // Handles:
    // { analysis: {...} }
    // { analysis: { analysis: {...} } }
    // direct {...}
    if (data?.analysis?.analysis) {
      return data.analysis.analysis;
    }

    if (data?.analysis) {
      return data.analysis;
    }

    return data;
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    setSelectedFile(file);
    setAnalysisResult(null);
    setErrorMessage("");

    if (file) {
      const previewUrl = URL.createObjectURL(file);
      setVideoPreview(previewUrl);
    }
  };

  const handleAnalyseVideo = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select a video first.");
      return;
    }

    setIsLoading(true);
    setErrorMessage("");
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyse", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      console.log("FULL BACKEND DATA:", data);

      if (!response.ok) {
        setErrorMessage(`Backend error: ${response.status}`);
        return;
      }

      if (data.error) {
        setErrorMessage(data.error);
        return;
      }

      if (data.analysis?.error) {
        setErrorMessage(data.analysis.error);
        return;
      }

      const normalisedResult = normaliseAnalysisResponse(data);

      console.log("NORMALISED RESULT:", normalisedResult);

      setAnalysisResult(normalisedResult);
    } catch (error) {
      console.error("Fetch error:", error);
      setErrorMessage(`Frontend error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const hasAnalysis = Boolean(analysisResult);

  return (
    <main className="app">
      <section className="hero">
        <p className="eyebrow">LiftLens</p>
        <h1>AI Fitness Form Feedback</h1>
        <p className="subtitle">
          Upload a squat video and get computer-vision based feedback on reps,
          depth, balance, and overall form.
        </p>
      </section>

      <section className="card upload-card">
        <h2>Upload squat video</h2>

        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="file-input"
        />

        {videoPreview && (
          <div className="video-preview">
            <video src={videoPreview} controls />
          </div>
        )}

        <button
          onClick={handleAnalyseVideo}
          disabled={isLoading}
          className="analyse-button"
        >
          {isLoading ? "Analysing..." : "Analyse Video"}
        </button>

        {errorMessage && <p className="error">{errorMessage}</p>}
      </section>

      {isLoading && (
        <section className="card">
          <h2>Analysing movement...</h2>
          <p>
            The backend is processing the video frame by frame using MediaPipe.
          </p>
        </section>
      )}

      {hasAnalysis && (
        <section className="results-grid">
          <div className="card stat-card">
            <p className="stat-label">Reps</p>
            <p className="stat-value">{analysisResult?.reps ?? "--"}</p>
          </div>

          <div className="card stat-card">
            <p className="stat-label">Score</p>
            <p className="stat-value">{analysisResult?.score ?? "--"}/100</p>
          </div>

          <div className="card stat-card">
            <p className="stat-label">Average Depth</p>
            <p className="stat-value">
              {analysisResult?.average_depth ?? "--"}°
            </p>
          </div>

          <div className="card stat-card">
            <p className="stat-label">Best Depth</p>
            <p className="stat-value">
              {analysisResult?.best_depth ?? "--"}°
            </p>
          </div>

          <div className="card stat-card">
            <p className="stat-label">Avg Imbalance</p>
            <p className="stat-value">
              {analysisResult?.average_imbalance ?? "--"}°
            </p>
          </div>

          <div className="card feedback-card">
            <h2>Feedback</h2>
            <ul>
              {analysisResult?.feedback?.length > 0 ? (
                analysisResult.feedback.map((item, index) => (
                  <li key={index}>{item}</li>
                ))
              ) : (
                <li>No feedback available.</li>
              )}
            </ul>
          </div>

          <div className="card rep-card">
            <h2>Rep Details</h2>

            {!analysisResult?.rep_details ||
            analysisResult.rep_details.length === 0 ? (
              <p>No rep details detected.</p>
            ) : (
              <div className="rep-list">
                {analysisResult.rep_details.map((rep) => (
                  <div key={rep.rep_number} className="rep-item">
                    <strong>Rep {rep.rep_number}</strong>
                    <span>{rep.depth_angle}°</span>
                    <span>{rep.depth_feedback}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card chart-card">
            <h2>Knee Angle Over Time</h2>

            {analysisResult?.angle_series &&
            analysisResult.angle_series.length > 0 ? (
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analysisResult.angle_series}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="frame"
                      label={{
                        value: "Frame",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      domain={[40, 190]}
                      label={{
                        value: "Angle",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="angle"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p>No angle data available.</p>
            )}
          </div>

          <div className="card debug-card">
            <h2>Debug Info</h2>
            <pre>{JSON.stringify(analysisResult?.debug ?? {}, null, 2)}</pre>
          </div>
        </section>
      )}
    </main>
  );
}

export default App;