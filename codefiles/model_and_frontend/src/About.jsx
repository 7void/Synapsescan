import { Heart, Zap, Shield } from "lucide-react";

function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900">About SynapseScan</h1>
            <button
              onClick={() => window.history.back()}
              className="text-gray-500 hover:text-gray-700 text-sm font-medium px-4 py-2 rounded-lg hover:bg-gray-100 transition"
            >
              ← Back to Home
            </button>
          </div>

          <div className="space-y-8 text-gray-700 leading-relaxed">
            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                <Heart className="w-6 h-6 text-purple-600 mr-3" />
                Our Mission
              </h2>
              <p className="text-lg">
                SynapseScan is an AI-powered medical analysis tool designed to assist healthcare professionals in the early detection and classification of ovarian cancer patterns. Our mission is to leverage cutting-edge artificial intelligence to support medical professionals in making more informed diagnostic decisions.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                <Zap className="w-6 h-6 text-purple-600 mr-3" />
                How It Works
              </h2>
              <p className="text-lg mb-4">
                Our advanced machine learning model has been trained on extensive medical imaging datasets to identify patterns and characteristics associated with ovarian cancer. The system analyzes uploaded medical images and provides:
              </p>
              <ul className="list-disc list-inside space-y-2 text-lg pl-4">
                <li>Automated classification of medical images</li>
                <li>Confidence scores for each prediction</li>
                <li>Quick analysis to support clinical decision-making</li>
                <li>User-friendly interface for healthcare professionals</li>
              </ul>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                <Shield className="w-6 h-6 text-purple-600 mr-3" />
                Technology & Accuracy
              </h2>
              <p className="text-lg">
                Built using state-of-the-art deep learning algorithms, SynapseScan employs convolutional neural networks specifically optimized for medical image analysis. Our model undergoes continuous validation and improvement to maintain high accuracy standards while ensuring reliable performance in clinical settings.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                For Healthcare Professionals
              </h2>
              <p className="text-lg">
                SynapseScan is designed as a diagnostic aid tool for qualified healthcare professionals. It is intended to supplement, not replace, clinical judgment and professional medical expertise. The system provides additional insights that can help inform diagnostic decisions when used in conjunction with standard medical practices.
              </p>
            </div>

            <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-xl border border-purple-100">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Commitment to Excellence
              </h2>
              <p className="text-lg">
                We are committed to advancing healthcare through responsible AI development. Our team continuously works to improve the accuracy, reliability, and accessibility of our diagnostic tools while maintaining the highest standards of medical ethics and patient privacy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;