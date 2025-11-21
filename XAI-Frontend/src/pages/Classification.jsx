import { useState } from "react";
import {
  Brain,
  Upload,
  Sparkles,
  FileText,
  Check,
  ChevronRight,
  X,
  Download,
} from "lucide-react";

const Classification = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedModel, setSelectedModel] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [selectedTechnique, setSelectedTechnique] = useState(null);
  const [results, setResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [modelPrediction, setModelPrediction] = useState(null);

  const models = [
    {
      id: "dog-cat",
      name: "Dog-Cat Classifier",
      description: "CNN model for classifying dog and cat images",
      icon: "🐕🐈",
      gradient: "from-purple-500 to-pink-500",
    },
    {
      id: "tb",
      name: "TB Diagnosis",
      description: "Medical imaging model for tuberculosis detection",
      icon: "🫁",
      gradient: "from-blue-500 to-cyan-500",
    },
  ];

  const techniques = [
    {
      id: "gradcam",
      name: "Grad-CAM",
      description:
        "Visualize which regions of the image influenced the decision",
      icon: "🔥",
      color: "blue",
    },
    {
      id: "shap",
      name: "SHAP",
      description: "Game-theory based feature importance visualization",
      icon: "📊",
      color: "green",
    },
  ];

  // UPDATED 5-STEP FLOW
  const steps = [
    { number: 1, name: "Select Model", icon: Brain },
    { number: 2, name: "Upload Image", icon: Upload },
    { number: 3, name: "Predict Output", icon: Check },
    { number: 4, name: "Choose XAI", icon: Sparkles },
    { number: 5, name: "View Results", icon: FileText },
  ];

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setResults({
      prediction: selectedModel === "dog-cat" ? "Dog" : "TB Positive",
      confidence: 94.3,
      visualization: uploadedImage,
      details: {
        primaryClass: selectedModel === "dog-cat" ? "Dog" : "TB Positive",
        secondaryClass: selectedModel === "dog-cat" ? "Cat" : "Normal",
        primaryProb: 94.3,
        secondaryProb: 5.7,
      },
    });

    setIsAnalyzing(false);
    setCurrentStep(5);
  };

  const resetFlow = () => {
    setCurrentStep(1);
    setSelectedModel(null);
    setUploadedImage(null);
    setSelectedTechnique(null);
    setResults(null);
    setModelPrediction(null);
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Progress Steps */}
        <div className="flex items-center justify-between mb-12">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.number;
            const isCompleted = currentStep > step.number;

            return (
              <div key={step.number} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ${
                      isCompleted
                        ? "bg-green-500 text-white"
                        : isActive
                        ? "bg-blue-600 text-white shadow-lg shadow-blue-500/50"
                        : "bg-gray-200 text-gray-400"
                    }`}
                  >
                    {isCompleted ? (
                      <Check className="w-6 h-6" />
                    ) : (
                      <Icon className="w-6 h-6" />
                    )}
                  </div>
                  <span
                    className={`mt-2 text-sm font-medium ${
                      isActive
                        ? "text-blue-600"
                        : isCompleted
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    {step.name}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-1 flex-1 mx-2 rounded transition-all duration-300 ${
                      isCompleted ? "bg-green-500" : "bg-gray-200"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* STEP 1 — MODEL SELECTION */}
          {currentStep === 1 && (
            <div className="p-8 md:p-12">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Choose Your Model
                </h2>
                <p className="text-gray-600">
                  Select the AI model for your diagnosis task
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model.id);
                      setCurrentStep(2);
                    }}
                    className={`group relative p-8 rounded-2xl border-2 transition-all duration-300 hover:shadow-2xl hover:scale-105 ${
                      selectedModel === model.id
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-200 hover:border-blue-300"
                    }`}
                  >
                    <div
                      className={`absolute inset-0 bg-linear-to-br ${model.gradient} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity`}
                    ></div>

                    <div className="relative">
                      <div className="text-5xl mb-4">{model.icon}</div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">
                        {model.name}
                      </h3>
                      <p className="text-sm text-gray-600 mb-4">
                        {model.description}
                      </p>
                      <div className="flex items-center text-blue-600 font-medium">
                        <span>Select Model</span>
                        <ChevronRight className="w-5 h-5 ml-1 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STEP 2 — UPLOAD IMAGE */}
          {currentStep === 2 && (
            <div className="p-8 md:p-12">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Upload Your Image
                </h2>
                <p className="text-gray-600">
                  Upload an image for{" "}
                  {selectedModel === "dog-cat"
                    ? "animal classification"
                    : "TB diagnosis"}
                </p>
              </div>

              <div className="max-w-2xl mx-auto">
                {!uploadedImage ? (
                  <label className="block">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                    <div className="border-4 border-dashed border-gray-300 rounded-2xl p-16 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition-all duration-300">
                      <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                      <p className="text-lg font-semibold text-gray-700 mb-2">
                        Click to upload or drag and drop
                      </p>
                      <p className="text-sm text-gray-500">
                        PNG, JPG, JPEG (Max 10MB)
                      </p>
                    </div>
                  </label>
                ) : (
                  <div className="space-y-6">
                    <div className="relative rounded-2xl overflow-hidden shadow-lg">
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="w-full h-96 object-cover"
                      />
                      <button
                        onClick={() => {
                          setUploadedImage(null);
                          setModelPrediction(null);
                        }}
                        className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>

                    <button
                      onClick={() => setCurrentStep(3)}
                      className="w-full bg-linear-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg hover:scale-105 transition-all duration-300"
                    >
                      Continue to Prediction
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* STEP 3 — PREDICT OUTPUT */}
          {currentStep === 3 && uploadedImage && (
            <div className="p-8 md:p-12">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Predict Model Output
                </h2>
                <p className="text-gray-600">
                  Generate the model’s prediction before running explainability.
                </p>
              </div>

              <div className="max-w-xl mx-auto space-y-6">
                <div className="rounded-2xl overflow-hidden shadow-lg">
                  <img
                    src={uploadedImage}
                    alt="Preview"
                    className="w-full h-72 object-cover"
                  />
                </div>

                {!modelPrediction ? (
                  <button
                    onClick={async () => {
                      setIsAnalyzing(true);

                      await new Promise((resolve) =>
                        setTimeout(resolve, 1500)
                      );

                      setModelPrediction(
                        selectedModel === "dog-cat" ? "Dog" : "TB Positive"
                      );

                      setIsAnalyzing(false);
                    }}
                    className="w-full bg-blue-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all duration-300 flex items-center justify-center space-x-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>Predicting...</span>
                      </>
                    ) : (
                      <>
                        <Check className="w-5 h-5" />
                        <span>Predict Output</span>
                      </>
                    )}
                  </button>
                ) : (
                  <div className="text-center space-y-4">
                    <div className="text-2xl font-bold text-gray-900">
                      Prediction:{" "}
                      <span className="text-blue-600">{modelPrediction}</span>
                    </div>

                    <button
                      onClick={() => setCurrentStep(4)}
                      className="w-full bg-green-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all duration-300"
                    >
                      Continue to XAI Selection
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* STEP 4 — XAI SELECTION */}
          {currentStep === 4 && (
            <div className="p-8 md:p-12">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Choose XAI Technique
                </h2>
                <p className="text-gray-600">
                  Select an explainability method to understand the prediction.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto mb-8">
                {techniques.map((technique) => (
                  <button
                    key={technique.id}
                    onClick={() => setSelectedTechnique(technique.id)}
                    className={`p-8 rounded-2xl border-2 transition-all duration-300 hover:shadow-xl ${
                      selectedTechnique === technique.id
                        ? `border-${technique.color}-500 bg-${technique.color}-50`
                        : "border-gray-200 hover:border-gray-300"
                    }`}
                  >
                    <div className="text-5xl mb-4">{technique.icon}</div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">
                      {technique.name}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {technique.description}
                    </p>

                    {selectedTechnique === technique.id && (
                      <div
                        className={`mt-4 flex items-center text-${technique.color}-600 font-medium`}
                      >
                        <Check className="w-5 h-5 mr-2" />
                        <span>Selected</span>
                      </div>
                    )}
                  </button>
                ))}
              </div>

              {selectedTechnique && (
                <div className="max-w-md mx-auto">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="w-full bg-linear-to-r from-green-500 to-emerald-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        <span>Run Analysis</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* STEP 5 — RESULTS PAGE */}
          {currentStep === 5 && results && (
            <div className="p-8 md:p-12">
              <div className="text-center mb-10">
                <div className="inline-block p-3 bg-green-100 rounded-full mb-4">
                  <Check className="w-8 h-8 text-green-600" />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Analysis Complete
                </h2>
                <p className="text-gray-600">
                  Here are your results with explainability insights.
                </p>
              </div>

              <div className="max-w-4xl mx-auto space-y-6">
                {/* Prediction Summary */}
                <div className="bg-linear-to-br from-blue-50 to-indigo-50 rounded-2xl p-8 border border-blue-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                      <h3 className="text-sm font-semibold text-gray-600 mb-2">
                        PREDICTION
                      </h3>
                      <p className="text-4xl font-bold text-gray-900 mb-4">
                        {results.prediction}
                      </p>

                      <h3 className="text-sm font-semibold text-gray-600 mb-2">
                        CONFIDENCE
                      </h3>
                      <div className="flex items-center space-x-3">
                        <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                          <div
                            className="bg-linear-to-r from-blue-500 to-indigo-600 h-full transition-all duration-1000"
                            style={{ width: `${results.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-2xl font-bold text-blue-600">
                          {results.confidence}%
                        </span>
                      </div>

                      <div className="mt-6 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-700">
                            {results.details.primaryClass}
                          </span>
                          <span className="font-semibold text-blue-600">
                            {results.details.primaryProb}%
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-700">
                            {results.details.secondaryClass}
                          </span>
                          <span className="font-semibold text-gray-600">
                            {results.details.secondaryProb}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-sm font-semibold text-gray-600 mb-2">
                        {selectedTechnique === "gradcam"
                          ? "GRAD-CAM VISUALIZATION"
                          : "SHAP ANALYSIS"}
                      </h3>
                      <div className="rounded-xl overflow-hidden shadow-lg">
                        <img
                          src={results.visualization}
                          alt="XAI Visualization"
                          className="w-full h-64 object-cover"
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Highlighted regions show areas influencing the model
                        decision.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <button className="flex items-center justify-center space-x-2 bg-linear-to-r from-green-500 to-emerald-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all duration-300">
                    <Download className="w-5 h-5" />
                    <span>Download Report</span>
                  </button>

                  <button
                    onClick={resetFlow}
                    className="flex items-center justify-center space-x-2 bg-gray-600 text-white py-4 rounded-xl font-semibold hover:bg-gray-700 hover:shadow-lg hover:scale-105 transition-all duration-300"
                  >
                    <ChevronRight className="w-5 h-5" />
                    <span>New Analysis</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Classification;
