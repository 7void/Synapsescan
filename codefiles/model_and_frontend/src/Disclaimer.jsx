import { AlertTriangle } from "lucide-react";

function DisclaimerPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 flex items-center">
              <AlertTriangle className="w-8 h-8 text-amber-500 mr-3" />
              Medical Disclaimer
            </h1>
            <button
              onClick={() => window.history.back()}
              className="text-gray-500 hover:text-gray-700 text-sm font-medium px-4 py-2 rounded-lg hover:bg-gray-100 transition"
            >
              ← Back to Home
            </button>
          </div>

          <div className="space-y-8 text-gray-700 leading-relaxed">
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-amber-800 mb-3">
                Important Notice
              </h2>
              <p className="text-amber-700 text-lg">
                This tool is for educational and research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Professional Use Only
              </h2>
              <p className="text-lg mb-4">
                SynapseScan is designed exclusively for use by qualified healthcare professionals with appropriate medical training and expertise. This tool should only be used by:
              </p>
              <ul className="list-disc list-inside space-y-2 text-lg pl-4">
                <li>Licensed medical doctors and specialists</li>
                <li>Certified medical imaging technicians</li>
                <li>Healthcare professionals with relevant training</li>
                <li>Medical researchers in authorized clinical settings</li>
              </ul>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Limitations and Accuracy
              </h2>
              <p className="text-lg mb-4">
                While our AI model has been trained on extensive datasets, it is important to understand the following limitations:
              </p>
              <ul className="list-disc list-inside space-y-2 text-lg pl-4">
                <li>The system may produce false positives or false negatives</li>
                <li>Results should always be verified through established medical protocols</li>
                <li>The tool is not FDA-approved for diagnostic purposes</li>
                <li>Image quality and type may affect accuracy</li>
                <li>Results should be interpreted in conjunction with clinical findings</li>
              </ul>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Clinical Decision Making
              </h2>
              <p className="text-lg">
                This AI tool is intended to assist and supplement clinical decision-making, not replace it. Healthcare professionals should always rely on their clinical judgment, patient history, additional diagnostic tests, and established medical guidelines when making treatment decisions.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Privacy and Data Security
              </h2>
              <p className="text-lg">
                We are committed to protecting patient privacy and medical data security. All uploaded images are processed locally and are not stored on our servers. Users are responsible for ensuring compliance with applicable privacy laws and regulations, including HIPAA requirements.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Emergency Situations
              </h2>
              <p className="text-lg">
                This tool should never be used in emergency medical situations. In case of a medical emergency, always contact emergency services immediately and follow established emergency medical protocols.
              </p>
            </div>

            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-red-800 mb-3">
                Liability Disclaimer
              </h2>
              <p className="text-red-700 text-lg">
                The creators and operators of SynapseScan assume no liability for any medical decisions made based on the output of this tool. Users accept full responsibility for their clinical decisions and patient care.
              </p>
            </div>

            <div className="text-center pt-6">
              <p className="text-gray-600 text-lg">
                By using SynapseScan, you acknowledge that you have read, understood, and agree to these terms and limitations.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DisclaimerPage;