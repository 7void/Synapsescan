// src/App.jsx
import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate
} from "react-router-dom";
import { ArrowRight, Upload, Zap, Heart } from "lucide-react";

import About from "./About";
import Disclaimer from "./Disclaimer";

function Home({ showClassifier, setShowClassifier, 
                image, preview, result, loading,
                handleImageChange, handleDrop, handleDragOver,
                classifyImage }) {
  const classNames = ['CC', 'EC', 'HGSC', 'LGSC', 'MC'];

  if (showClassifier) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white px-4">
        <div className="w-full max-w-md bg-gray-100 border border-gray-300 rounded-2xl shadow-xl p-6 space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-semibold text-gray-900">
              Ovarian Cancer Classifier
            </h1>
            <button
              onClick={() => setShowClassifier(false)}
              className="text-gray-500 hover:text-gray-700 text-sm"
            >
              ← Back
            </button>
          </div>

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className="flex flex-col items-center justify-center border-2 border-dashed border-gray-400 rounded-xl p-6 cursor-pointer hover:border-gray-600 transition"
            onClick={() => document.getElementById("fileInput").click()}
          >
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                className="w-48 h-48 object-contain rounded-md"
              />
            ) : (
              <div className="text-center text-gray-600">
                <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-sm">Drag & drop an image here</p>
                <p className="text-sm">or click to upload</p>
              </div>
            )}
          </div>

          <input
            id="fileInput"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageChange}
          />

          {image && (
            <div className="text-center space-y-4">
              <p className="text-sm text-gray-700">{image.name}</p>
              <button
                onClick={classifyImage}
                disabled={loading}
                className="px-4 py-2 rounded-md bg-black text-white hover:bg-gray-800 transition disabled:opacity-50"
              >
                {loading ? "Classifying..." : "Classify Image"}
              </button>

              {result && (
                <div className="mt-4 p-4 bg-white shadow rounded text-left">
                  <p>
                    <strong>Prediction:</strong> {classNames[result.classIndex]}
                  </p>
                  <p>
                    <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="px-6 py-4">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Heart className="w-8 h-8 text-purple-600" />
            <span className="text-xl font-bold text-gray-900">SynapseScan</span>
          </div>
          <nav className="hidden md:flex space-x-8 text-gray-600">
            <Link to="/about">About</Link>
            <Link to="/disclaimer">Disclaimer</Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-purple-100 text-purple-700 text-sm font-medium mb-8">
            <Zap className="w-4 h-4 mr-2" />
            AI-Powered Medical Analysis
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
            Ovarian Cancer
            <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent"> Classifier</span>
          </h1>
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
            Advanced AI technology to assist healthcare professionals in
            analyzing medical images for early detection and classification
            of ovarian cancer patterns.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => setShowClassifier(true)}
              className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all duration-300 flex items-center justify-center"
            >
              Start Classification
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function AppWrapper() {
  const [showClassifier, setShowClassifier] = useState(false);
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setResult(null);
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setShowClassifier(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    setResult(null);
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setShowClassifier(true);
  };

  const handleDragOver = (e) => e.preventDefault();

  const classifyImage = async () => {
    if (!image) return;
    setLoading(true);
    const form = new FormData();
    form.append("file", image);
    try {
      const res = await fetch("http://localhost:5000/classify", {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to classify image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Routes>
      <Route
        path="/"
        element={
          <Home
            showClassifier={showClassifier}
            setShowClassifier={setShowClassifier}
            image={image}
            preview={preview}
            result={result}
            loading={loading}
            handleImageChange={handleImageChange}
            handleDrop={handleDrop}
            handleDragOver={handleDragOver}
            classifyImage={classifyImage}
          />
        }
      />
      <Route path="/about" element={<About />} />
      <Route path="/disclaimer" element={<Disclaimer />} />
      <Route path="*" element={<Home {...{ showClassifier, setShowClassifier, image, preview, result, loading, handleImageChange, handleDrop, handleDragOver, classifyImage }} />} />
    </Routes>
  );
}

export default function App() {
  return (
    <Router>
      <AppWrapper />
    </Router>
  );
}
