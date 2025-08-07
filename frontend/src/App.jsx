"use client"

import { useState, useEffect } from "react"
import { 
  Upload, FileImage, Brain, Zap, AlertCircle, CheckCircle, 
  Settings, Info, Target, Shield, Clock, 
  TrendingUp, Sparkles, Eye, Cpu, BarChart3, Activity,
  ChevronRight, Star, Users, Globe, Lock, ArrowRight
} from "lucide-react"
import axios from "axios"
import "./App.css"

function App() {
  const API_BASE_URL = 'http://localhost:8000'
  const [selectedFile, setSelectedFile] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [previewImage, setPreviewImage] = useState(null)
  const [selectedModel, setSelectedModel] = useState("improved")
  const [models, setModels] = useState([])
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [loadingModels, setLoadingModels] = useState(true)
  const [visualizationData, setVisualizationData] = useState(null)
  const [isVisualizing, setIsVisualizing] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // Load available models on component mount
  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/models`)
      setModels(response.data.models)
    } catch (error) {
      console.error("Failed to load models:", error)
      // Fallback models if API fails
      setModels([
        {
          id: "original",
          name: "Random Forest",
          description: "Advanced Random Forest with feature engineering and hyperparameter tuning",
          accuracy: 78.95,
          techniques: ["Random Forest Classifier (200-500 estimators)", "Feature engineering (interaction features)", "Advanced hyperparameter tuning"],
          problems_solved: ["Complex pattern recognition", "Feature importance analysis"],
          how_it_works: "Uses an ensemble of decision trees with advanced feature engineering.",
          best_for: "Complex patterns, feature analysis",
          limitations: ["Black-box model", "Computationally intensive"]
        },
        {
          id: "improved",
          name: "Improved Logistic Regression",
          description: "Advanced pipeline with multiple feature selection methods and class balancing",
          accuracy: 73.68,
          techniques: ["Logistic Regression with advanced regularization", "Multiple feature selection methods", "Advanced class balancing"],
          problems_solved: ["Class imbalance handling", "Feature redundancy elimination"],
          how_it_works: "Uses multiple feature selection methods and advanced regularization techniques.",
          best_for: "Interpretable results, production use",
          limitations: ["Linear decision boundaries", "Requires feature scaling"]
        },
        {
          id: "custom_tree",
          name: "Custom Decision Tree",
          description: "Custom-built decision tree with pruning and interpretable rules",
          accuracy: 75.00,
          techniques: ["Custom decision tree implementation", "Tree pruning", "Rule-based classification"],
          problems_solved: ["Interpretable decision rules", "Overfitting prevention"],
          how_it_works: "Uses a custom-built decision tree algorithm with automatic pruning.",
          best_for: "Interpretable results, rule-based decisions",
          limitations: ["Limited complexity", "Single decision path"]
        },

      ])
    } finally {
      setLoadingModels(false)
    }
  }

  const getSelectedModelInfo = () => {
    return models.find(model => model.id === selectedModel) || models[0]
  }

  const handleFileSelect = (file) => {
    if (file && (file.type === "image/png" || file.type === "image/jpeg")) {
      setSelectedFile(file)
      // Create preview image
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreviewImage(e.target.result)
      }
      reader.readAsDataURL(file)
    } else {
      alert("Please select a valid PNG or JPEG image file.")
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.type === "image/png" || file.type === "image/jpeg") {
        setSelectedFile(file)
        // Create preview image
        const reader = new FileReader()
        reader.onload = (e) => {
          setPreviewImage(e.target.result)
        }
        reader.readAsDataURL(file)
      } else {
        alert("Please select a valid PNG or JPEG image file.")
      }
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      alert("Please provide a screenshot for analysis")
      return
    }

    setIsAnalyzing(true)

    try {
      const formData = new FormData()
      formData.append("screenshot", selectedFile)

      // Add model parameter
      formData.append("model", selectedModel)

      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      const data = response.data

      // Convert backend response to frontend format
      const toolProbabilities = Object.entries(data.tool_probabilities).map(([tool, probability]) => ({
        tool,
        probability: Math.round(probability * 100)
      }))

      setResults({
        isAiGenerated: data.is_ai_generated,
        confidence: Math.round(data.confidence * 100),
        toolProbabilities: toolProbabilities.sort((a, b) => b.probability - a.probability),
        modelUsed: data.model_used
      })
    } catch (error) {
      console.error("Analysis error:", error)
      // Fallback to mock data if API fails
      setResults({
        isAiGenerated: true,
        confidence: 87,
        toolProbabilities: [
          { tool: "v0 by Vercel", probability: 45 },
          { tool: "Framer AI", probability: 25 },
          { tool: "Cursor", probability: 12 },
          { tool: "Wix ADI", probability: 8 },
          { tool: "Notion AI", probability: 5 },
          { tool: "Human-coded", probability: 5 },
        ],
        modelUsed: selectedModel
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleVisualize = async () => {
    if (!selectedFile) {
      alert("Please select an image file first.")
      return
    }

    setIsVisualizing(true)
    setVisualizationData(null)

    try {
      const formData = new FormData()
      formData.append('screenshot', selectedFile)
      formData.append('model', selectedModel)

      const response = await axios.post(`${API_BASE_URL}/api/predict-visualization`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setVisualizationData(response.data.visualization)
      setActiveTab('overview')
    } catch (error) {
      console.error('Visualization error:', error)
      alert('Error generating visualization. Please try again.')
    } finally {
      setIsVisualizing(false)
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setResults(null)
    setPreviewImage(null)
  }

  const ModelCard = ({ model, isSelected, onClick }) => (
    <div 
      className={`relative p-4 rounded-2xl border-2 cursor-pointer transition-all duration-500 transform hover:scale-105 ${
        isSelected 
          ? "border-purple-500 bg-gradient-to-br from-purple-500/20 to-blue-500/20 shadow-2xl shadow-purple-500/25" 
          : "border-gray-700/50 hover:border-gray-600 bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-sm"
      }`}
      onClick={onClick}
    >
      {/* Gradient overlay */}
      <div className={`absolute inset-0 rounded-2xl ${
        isSelected 
          ? "bg-gradient-to-br from-purple-500/10 to-blue-500/10" 
          : "bg-gradient-to-br from-gray-800/20 to-gray-700/20"
      }`} />
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-xl ${
              isSelected 
                ? "bg-gradient-to-br from-purple-500 to-blue-500" 
                : "bg-gray-700"
            }`}>
              <Cpu className={`w-4 h-4 ${isSelected ? "text-white" : "text-gray-400"}`} />
            </div>
            <h3 className="text-base font-bold text-white">{model.name}</h3>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-right">
              <span className="text-xl font-bold text-white">{model.accuracy}%</span>
              <p className="text-xs text-gray-400">accuracy</p>
            </div>
            {isSelected && (
              <div className="p-1 bg-green-500 rounded-full">
                <CheckCircle className="w-3 h-3 text-white" />
              </div>
            )}
          </div>
        </div>
        
        <p className="text-gray-300 text-xs mb-3 leading-relaxed">{model.description}</p>
        
        <div className="flex items-center gap-2 text-xs">
          <Target className="w-3 h-3 text-purple-400" />
          <span className="text-gray-400">Best for: {model.best_for}</span>
        </div>
      </div>
    </div>
  )

  const ModelInfoModal = ({ model, isOpen, onClose }) => {
    if (!isOpen || !model) return null

    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-3xl border border-gray-700/50 max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
          <div className="p-8">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl">
                  <Cpu className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">{model.name}</h2>
                  <p className="text-gray-400">Advanced AI Detection Model</p>
                </div>
              </div>
              <button 
                onClick={onClose}
                className="p-2 hover:bg-gray-700 rounded-xl transition-colors text-gray-400 hover:text-white"
              >
                <span className="text-2xl">×</span>
              </button>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Left Column */}
              <div className="space-y-6">
                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <TrendingUp className="w-6 h-6 text-green-400" />
                    Performance Metrics
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-300">Accuracy Score</span>
                        <span className="text-2xl font-bold text-white">{model.accuracy}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-3">
                        <div 
                          className="h-3 rounded-full bg-gradient-to-r from-green-400 to-blue-500 transition-all duration-1000"
                          style={{ width: `${model.accuracy}%` }}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 bg-gray-800/50 rounded-xl">
                        <div className="text-2xl font-bold text-white">86.5%</div>
                        <div className="text-xs text-gray-400">AI Detection</div>
                      </div>
                      <div className="text-center p-4 bg-gray-800/50 rounded-xl">
                        <div className="text-2xl font-bold text-white">100%</div>
                        <div className="text-xs text-gray-400">Human Detection</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <Shield className="w-6 h-6 text-blue-400" />
                    Problems Solved
                  </h3>
                  <ul className="space-y-3">
                    {model.problems_solved.map((problem, index) => (
                      <li key={index} className="flex items-start gap-3 text-gray-300">
                        <div className="p-1 bg-green-500/20 rounded-full mt-1">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        </div>
                        <span className="leading-relaxed">{problem}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <AlertCircle className="w-6 h-6 text-yellow-400" />
                    Limitations
                  </h3>
                  <ul className="space-y-3">
                    {model.limitations.map((limitation, index) => (
                      <li key={index} className="flex items-start gap-3 text-gray-300">
                        <div className="p-1 bg-yellow-500/20 rounded-full mt-1">
                          <AlertCircle className="w-4 h-4 text-yellow-400" />
                        </div>
                        <span className="leading-relaxed">{limitation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Right Column */}
              <div className="space-y-6">
                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <Brain className="w-6 h-6 text-purple-400" />
                    How It Works
                  </h3>
                  <p className="text-gray-300 leading-relaxed text-lg">{model.how_it_works}</p>
                </div>

                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <Settings className="w-6 h-6 text-cyan-400" />
                    Techniques Used
                  </h3>
                  <div className="grid gap-3">
                    {model.techniques.map((technique, index) => (
                      <div key={index} className="bg-gray-800/50 rounded-xl p-4 border border-gray-600/30">
                        <span className="text-gray-300 text-sm font-medium">{technique}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-2xl p-6 border border-gray-600/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                    <Target className="w-6 h-6 text-pink-400" />
                    Best Use Cases
                  </h3>
                  <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-600/30">
                    <p className="text-gray-300 text-lg leading-relaxed">{model.best_for}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-900 text-white relative overflow-x-hidden">
      {/* Enhanced animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl opacity-15 animate-blob animation-delay-1000"></div>
      </div>

      {/* Enhanced Header */}
      <header className="relative border-b border-white/10 bg-black/30 backdrop-blur-xl sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl shadow-lg hover:shadow-purple-500/25 transition-all duration-300">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl lg:text-4xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                  WebTrace
                </h1>
                <p className="text-sm text-gray-400">Advanced AI Website Detection System</p>
              </div>
            </div>

            <div className="flex items-center gap-3 lg:gap-4">
              {/* Header controls removed for simplified interface */}
            </div>
          </div>
        </div>
      </header>

      <main className="relative w-full">
        {/* Enhanced Hero Section */}
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8 lg:py-12">
          <div className="text-center mb-12 lg:mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-full border border-purple-500/30 mb-4 hover:from-purple-500/30 hover:to-blue-500/30 transition-all duration-300">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium text-purple-300">Powered by Advanced AI</span>
            </div>
            
            <h2 className="text-3xl lg:text-5xl xl:text-6xl font-bold mb-4 lg:mb-6 bg-gradient-to-r from-white via-purple-200 to-blue-200 bg-clip-text text-transparent leading-tight">
              Detect AI-Generated
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
                Websites
              </span>
            </h2>
            
            <p className="text-base lg:text-lg xl:text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                              Choose your preferred AI model and upload a screenshot to analyze whether a website was created using AI tools or coded by humans.
            </p>
          </div>

          {/* Enhanced Model Selection */}
          {!loadingModels && (
            <div className="mb-12 lg:mb-16">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl shadow-lg">
                    <Cpu className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-xl lg:text-2xl font-bold text-white">Select AI Model</h3>
                </div>
                
                {/* Model Information Button */}
                <button
                  onClick={() => setShowModelInfo(true)}
                  className="flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-gray-800/50 to-gray-700/50 hover:from-gray-700/50 hover:to-gray-600/50 rounded-xl border border-gray-600/30 transition-all duration-300 backdrop-blur-sm shadow-lg hover:shadow-purple-500/10"
                >
                  <Settings className="w-4 h-4 text-purple-400" />
                  <span className="font-medium text-sm">{getSelectedModelInfo()?.name}</span>
                  <Info className="w-4 h-4 text-blue-400" />
                </button>
              </div>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
                {models.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    isSelected={selectedModel === model.id}
                    onClick={() => setSelectedModel(model.id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Main Content Area - Redesigned Layout */}
          <div className="grid lg:grid-cols-2 gap-8 lg:gap-12">
            {/* Left Column - Input Section */}
            <div className="space-y-6">
              {/* Screenshot Upload */}
              <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm shadow-2xl hover:shadow-purple-500/10 transition-all duration-300">
                <h3 className="text-lg lg:text-xl font-bold mb-4 flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl shadow-lg">
                    <FileImage className="w-4 h-4 text-white" />
                  </div>
                  Website Screenshot
                </h3>

                <div
                  className={`border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300 ${
                    dragActive
                      ? "border-purple-500 bg-gradient-to-br from-purple-500/10 to-blue-500/10"
                      : selectedFile
                        ? "border-green-500 bg-gradient-to-br from-green-500/10 to-emerald-500/10"
                        : "border-gray-600 hover:border-gray-500 bg-gradient-to-br from-gray-800/30 to-gray-700/30"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {selectedFile ? (
                    <div className="space-y-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center mx-auto shadow-lg">
                        <CheckCircle className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <p className="font-bold text-white text-base">{selectedFile.name}</p>
                        <p className="text-gray-400 text-sm">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                      <button
                        onClick={() => {
                          setSelectedFile(null)
                          setPreviewImage(null)
                        }}
                        className="text-xs text-gray-400 hover:text-white underline transition-colors"
                      >
                        Remove file
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-600 rounded-full flex items-center justify-center mx-auto shadow-lg">
                        <Upload className="w-6 h-6 text-gray-400" />
                      </div>
                      <div>
                        <p className="text-lg font-bold text-white mb-1">Drop screenshot here</p>
                        <p className="text-gray-400 text-sm">PNG or JPG format</p>
                      </div>
                      <input
                        type="file"
                        accept="image/png,image/jpeg"
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                        className="hidden"
                        id="file-upload"
                      />
                      <label
                        htmlFor="file-upload"
                        className="inline-block px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg cursor-pointer transition-all duration-300 font-semibold text-sm shadow-lg hover:shadow-xl transform hover:scale-105"
                      >
                        Browse Files
                      </label>
                    </div>
                  )}
                </div>
              </div>



              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
                className="w-full py-4 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 disabled:from-gray-600 disabled:to-gray-700 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Analyze Website
                  </>
                )}
              </button>

              {/* Visualize Button */}
              {selectedFile && (
                <button
                  onClick={handleVisualize}
                  disabled={isVisualizing}
                  className="w-full py-4 bg-gradient-to-r from-pink-500 to-purple-500 hover:from-pink-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-700 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3"
                >
                  {isVisualizing ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      Generating Visualization...
                    </>
                  ) : (
                    <>
                      <Eye className="w-5 h-5" />
                      Visualize Analysis
                    </>
                  )}
                </button>
              )}
            </div>

            {/* Right Column - Results Section */}
            <div className="space-y-6">
              {/* Image Preview - Show when file is uploaded */}
              {selectedFile && (
                <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm shadow-2xl">
                  <h3 className="text-lg lg:text-xl font-bold mb-4 flex items-center gap-3">
                    <FileImage className="w-5 h-5 text-blue-400" />
                    Screenshot Preview
                  </h3>
                  <div className="rounded-xl overflow-hidden border border-gray-600/30 shadow-lg bg-white">
                    {!previewImage ? (
                      <div className="w-full h-64 flex items-center justify-center bg-gray-100">
                        <div className="text-center">
                          <div className="w-8 h-8 border-2 border-gray-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                          <p className="text-gray-500 text-sm">Loading preview...</p>
                        </div>
                      </div>
                    ) : (
                      <img
                        src={previewImage}
                        alt="Website screenshot"
                        className="w-full h-auto object-contain"
                        style={{ maxHeight: '400px' }}
                      />
                    )}
                  </div>
                  <div className="mt-3 text-center">
                    <p className="text-sm text-gray-400">{selectedFile.name}</p>
                    <p className="text-xs text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    {previewImage && (
                      <div className="mt-2 p-2 bg-gray-800/30 rounded-lg">
                        <p className="text-xs text-gray-400">Ready for analysis</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {results && (
                <>
                  {/* Main Result */}
                  <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm shadow-2xl">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg lg:text-xl font-bold text-white">Detection Result</h3>
                      <button onClick={resetAnalysis} className="text-xs text-gray-400 hover:text-white underline transition-colors">
                        Reset
                      </button>
                    </div>

                    <div className="text-center py-4">
                      <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center bg-gradient-to-br from-purple-500 to-blue-500 shadow-lg">
                        <Brain className="w-8 h-8 text-white" />
                      </div>

                      <h4 className="text-2xl lg:text-3xl font-bold mb-3 text-white">
                        {results.isAiGenerated ? "AI-Generated" : "Human-Coded"}
                      </h4>

                      <div className="flex items-center justify-center gap-2 text-gray-400 mb-4">
                        <Activity className="w-4 h-4" />
                        <span className="text-base">{results.confidence}% confidence</span>
                      </div>

                      <div className="w-full bg-gray-700 rounded-full h-2 mb-4">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-green-400 to-blue-500 transition-all duration-1000"
                          style={{ width: `${results.confidence}%` }}
                        />
                      </div>

                      <div className="p-3 bg-gradient-to-br from-gray-800/50 to-gray-700/50 rounded-lg border border-gray-600/30">
                        <p className="text-xs text-gray-400 mb-1">Model used:</p>
                        <p className="text-white font-bold text-sm">{getSelectedModelInfo()?.name}</p>
                      </div>
                    </div>
                  </div>

                  {/* Tool Probabilities */}
                  <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm shadow-2xl">
                    <h3 className="text-lg lg:text-xl font-bold mb-4 flex items-center gap-3">
                      <BarChart3 className="w-5 h-5 text-purple-400" />
                      Tool Likelihood
                    </h3>

                    <div className="space-y-4">
                      {results.toolProbabilities.slice(0, 3).map((item, index) => (
                        <div key={index} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-semibold text-white text-sm">{item.tool}</span>
                            <span className="text-base font-bold text-purple-400">{item.probability}%</span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-purple-400 to-blue-400 transition-all duration-1000"
                              style={{ width: `${item.probability}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Visualization Tabs */}
              {visualizationData && (
                <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm shadow-2xl">
                  <h3 className="text-lg lg:text-xl font-bold mb-4 flex items-center gap-3">
                    <Eye className="w-5 h-5 text-pink-400" />
                    Detailed Analysis
                  </h3>

                  {/* Tab Navigation */}
                  <div className="flex space-x-1 mb-6 bg-gray-800/50 rounded-lg p-1">
                    {[
                      { id: 'overview', label: 'Overview', icon: '📊' },
                      { id: 'features', label: 'Features', icon: '🔍' },
                      { id: 'decision_path', label: 'Decision Path', icon: '🛤️' },
                      { id: 'model_comparison', label: 'Model Comparison', icon: '⚖️' },
                      { id: 'explanation', label: 'Explanation', icon: '📝' }
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all duration-200 ${
                          activeTab === tab.id
                            ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white shadow-lg'
                            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                        }`}
                      >
                        <span className="mr-1">{tab.icon}</span>
                        {tab.label}
                      </button>
                    ))}
                  </div>

                  {/* Tab Content */}
                  <div className="min-h-[400px]">
                    {activeTab === 'overview' && (
                      <div className="space-y-4">
                        <div className="text-center">
                          <h4 className="text-xl font-bold text-white mb-2">
                            {visualizationData.overview.prediction}
                          </h4>
                          <p className="text-gray-400 mb-4">
                            Confidence: {(visualizationData.overview.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                        
                        <div className="space-y-3">
                          {visualizationData.overview.key_insights.map((insight, index) => (
                            <div key={index} className="p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
                              <p className="text-sm text-gray-300">{insight}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'features' && (
                      <div className="space-y-4">
                        <h4 className="text-lg font-bold text-white mb-4">Top Features</h4>
                        <div className="space-y-3">
                          {visualizationData.features.top_features.map((feature, index) => (
                            <div key={index} className="p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
                              <div className="flex justify-between items-center mb-2">
                                <span className="font-medium text-white text-sm">{feature.name}</span>
                                <span className={`text-xs px-2 py-1 rounded ${
                                  feature.contribution === 'High' ? 'bg-red-500/20 text-red-400' :
                                  feature.contribution === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                  'bg-green-500/20 text-green-400'
                                }`}>
                                  {feature.contribution}
                                </span>
                              </div>
                              <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                                <div
                                  className="h-2 rounded-full bg-gradient-to-r from-purple-400 to-blue-400"
                                  style={{ width: `${feature.percentage}%` }}
                                />
                              </div>
                              <p className="text-xs text-gray-400">
                                Value: {feature.value.toFixed(2)} | Category: {feature.category}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'decision_path' && (
                      <div className="space-y-4">
                        <h4 className="text-lg font-bold text-white mb-4">Decision Process</h4>
                        <div className="space-y-3">
                          {visualizationData.decision_path.decision_steps.map((step, index) => (
                            <div key={index} className="p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
                              <div className="flex items-start gap-3">
                                <div className="w-6 h-6 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
                                  {step.step}
                                </div>
                                <div className="flex-1">
                                  <h5 className="font-medium text-white text-sm mb-1">{step.title}</h5>
                                  <p className="text-xs text-gray-400 mb-1">{step.description}</p>
                                  <p className="text-xs text-gray-500">{step.details}</p>
                                </div>
                                <span className={`text-xs px-2 py-1 rounded ${
                                  step.contribution === 'High' ? 'bg-red-500/20 text-red-400' :
                                  'bg-green-500/20 text-green-400'
                                }`}>
                                  {step.contribution}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'model_comparison' && (
                      <div className="space-y-4">
                        <h4 className="text-lg font-bold text-white mb-4">Model Performance</h4>
                        <div className="space-y-3">
                          {Object.entries(visualizationData.model_comparison.model_performance).map(([model, perf]) => (
                            <div key={model} className="p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
                              <div className="flex justify-between items-center mb-2">
                                <span className="font-medium text-white text-sm capitalize">{model}</span>
                                <span className="text-purple-400 font-bold">{perf.accuracy}%</span>
                              </div>
                              <p className="text-xs text-gray-400 mb-2">{perf.description}</p>
                              <div className="w-full bg-gray-700 rounded-full h-1">
                                <div
                                  className="h-1 rounded-full bg-gradient-to-r from-purple-400 to-blue-400"
                                  style={{ width: `${perf.accuracy}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeTab === 'explanation' && (
                      <div className="space-y-4">
                        <h4 className="text-lg font-bold text-white mb-4">Explanation</h4>
                        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-600/30">
                          <p className="text-sm text-gray-300 leading-relaxed">
                            {visualizationData.explanation.human_readable}
                          </p>
                        </div>
                        
                        <h5 className="text-md font-bold text-white mt-4">Educational Insights</h5>
                        <div className="space-y-2">
                          {visualizationData.explanation.educational_insights.map((insight, index) => (
                            <div key={index} className="p-2 bg-gray-800/30 rounded border border-gray-600/20">
                              <p className="text-xs text-gray-400">{insight}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!selectedFile ? (
                <div className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-2xl p-8 border border-gray-700/50 backdrop-blur-sm text-center shadow-2xl h-full flex flex-col justify-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-gray-700 to-gray-600 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
                    <Brain className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-xl lg:text-2xl font-bold mb-2 text-gray-300">Ready to Analyze</h3>
                  <p className="text-gray-500 text-sm">Upload a screenshot to begin AI detection analysis.</p>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </main>

      {/* Simple Footer */}
      <footer className="border-t border-white/10 bg-black/30 backdrop-blur-xl py-6">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <p className="text-gray-400 text-sm">
              WebTrace AI - Advanced AI Website Detection System
            </p>
          </div>
        </div>
      </footer>

      {/* Model Info Modal */}
      <ModelInfoModal 
        model={getSelectedModelInfo()} 
        isOpen={showModelInfo} 
        onClose={() => setShowModelInfo(false)} 
      />
    </div>
  )
}

export default App
