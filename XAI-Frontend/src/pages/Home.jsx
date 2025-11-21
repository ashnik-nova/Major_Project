import React, { useState, useEffect } from 'react';
import { Brain, Eye, Zap, Shield, TrendingUp, ChevronDown } from 'lucide-react';

export default function InterpretableDLHome() {
  const [scrollY, setScrollY] = useState(0);
  const [activeSection, setActiveSection] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const features = [
    {
      icon: <Eye className="w-8 h-8" />,
      title: "Visual Explanations",
      description: "Grad-CAM highlights critical image regions that influence model decisions, making AI transparent and trustworthy."
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Feature Attribution",
      description: "SHAP provides precise feature-level importance scores, revealing how each input contributes to predictions."
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Custom CNN Architecture",
      description: "Built from scratch with optimized layers for superior performance and interpretability across diverse datasets."
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "Bias Detection",
      description: "Systematic analysis identifies and mitigates algorithmic biases, ensuring fair and ethical AI deployment."
    }
  ];

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-600 via-blue-500 to-cyan-400 text-gray-800 overflow-x-hidden">
      {/* Hero Section */}
      <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div 
            className="absolute w-96 h-96 bg-white/10 rounded-full blur-3xl -top-48 -left-48"
            style={{ transform: `translate(${scrollY * 0.1}px, ${scrollY * 0.1}px)` }}
          />
          <div 
            className="absolute w-96 h-96 bg-cyan-300/20 rounded-full blur-3xl top-1/2 -right-48"
            style={{ transform: `translate(${-scrollY * 0.15}px, ${scrollY * 0.05}px)` }}
          />
          <div 
            className="absolute w-72 h-72 bg-blue-400/20 rounded-full blur-3xl bottom-0 left-1/3"
            style={{ transform: `translate(${scrollY * 0.08}px, ${-scrollY * 0.08}px)` }}
          />
        </div>

        {/* Hero Content */}
        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          <div className="mb-8 inline-flex items-center gap-3 bg-white/20 backdrop-blur-md px-6 py-3 rounded-full border border-white/30 animate-fadeIn">
            <Brain className="w-6 h-6 text-white" />
            <span className="text-white font-medium">Explainable AI Research</span>
          </div>
          
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 leading-tight animate-slideUp">
            Interpretable Deep Learning
            <span className="block mt-2 bg-linear-to-r from-cyan-200 to-white bg-clip-text text-transparent">
              for Image Classification
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-3xl mx-auto font-light animate-slideUp" style={{ animationDelay: '0.2s' }}>
            Unveiling the black box with Grad-CAM and SHAP — Making AI decisions transparent, trustworthy, and explainable
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-slideUp" style={{ animationDelay: '0.4s' }}>
            <button className="group bg-white text-blue-600 px-8 py-4 rounded-full font-semibold text-lg hover:shadow-2xl hover:scale-105 transition-all duration-300">
              Explore Research
              <ChevronDown className="inline-block ml-2 w-5 h-5 group-hover:translate-y-1 transition-transform" />
            </button>
            <button className="bg-white/10 backdrop-blur-md text-white border-2 border-white/30 px-8 py-4 rounded-full font-semibold text-lg hover:bg-white/20 hover:scale-105 transition-all duration-300">
              View Demo
            </button>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-bounce">
          <ChevronDown className="w-8 h-8 text-white/70" />
        </div>
      </div>

      {/* Problem Statement Section */}
      <div className="relative py-24 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="inline-block bg-blue-100 text-blue-600 px-4 py-2 rounded-full text-sm font-semibold mb-4">
              THE CHALLENGE
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Breaking the Black Box Barrier
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Deep learning models excel at predictions but fail to explain their reasoning. In critical domains like healthcare and autonomous systems, this opacity creates trust deficits and regulatory challenges.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mt-12">
            <div className="bg-linear-to-br from-red-50 to-orange-50 p-8 rounded-2xl border border-red-100">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">The Problem</h3>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start gap-3">
                  <span className="text-red-500 font-bold">×</span>
                  <span>Neural networks operate as opaque black boxes</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-500 font-bold">×</span>
                  <span>No systematic comparison of interpretability methods</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-500 font-bold">×</span>
                  <span>Hidden algorithmic biases remain undetected</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-500 font-bold">×</span>
                  <span>Practitioners lack guidance for method selection</span>
                </li>
              </ul>
            </div>

            <div className="bg-linear-to-br from-blue-50 to-cyan-50 p-8 rounded-2xl border border-blue-100">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Our Solution</h3>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 font-bold">✓</span>
                  <span>Systematic comparative analysis framework</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 font-bold">✓</span>
                  <span>Quantitative fidelity and robustness metrics</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 font-bold">✓</span>
                  <span>Bias detection and mitigation techniques</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 font-bold">✓</span>
                  <span>Custom CNN architecture optimization</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative py-24 bg-linear-to-br from-gray-50 to-blue-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="inline-block bg-blue-100 text-blue-600 px-4 py-2 rounded-full text-sm font-semibold mb-4">
              KEY FEATURES
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Powered by Advanced Techniques
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, idx) => (
              <div
                key={idx}
                className="group bg-white p-8 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border border-gray-100"
              >
                <div className="w-16 h-16 bg-linear-to-br from-blue-500 to-cyan-400 rounded-2xl flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Objectives Section */}
      <div className="relative py-24 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="inline-block bg-blue-100 text-blue-600 px-4 py-2 rounded-full text-sm font-semibold mb-4">
              RESEARCH OBJECTIVES
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Our Mission
            </h2>
          </div>

          <div className="space-y-6">
            {[
              "Systematically compare Grad-CAM and SHAP interpretability techniques on custom CNN architectures",
              "Quantitatively evaluate fidelity, robustness, and computational efficiency across both methods",
              "Investigate and identify algorithmic biases in trained models and develop mitigation strategies",
              "Provide evidence-based guidance for practitioners deploying explainable AI systems"
            ].map((objective, idx) => (
              <div
                key={idx}
                className="flex items-start gap-6 bg-linear-to-r from-blue-50 to-transparent p-6 rounded-xl hover:from-blue-100 transition-colors duration-300"
              >
                <div className="shrink-0 w-10 h-10 bg-linear-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center text-white font-bold text-lg">
                  {idx + 1}
                </div>
                <p className="text-lg text-gray-700 pt-1">{objective}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="relative py-24 bg-linear-to-br from-blue-600 via-blue-500 to-cyan-400 overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-0 w-full h-full" style={{
            backgroundImage: 'radial-gradient(circle, white 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }} />
        </div>
        
        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Explore Explainable AI?
          </h2>
          <p className="text-xl text-white/90 mb-10">
            Dive into our research and discover how we're making deep learning transparent and trustworthy
          </p>
          <button className="bg-white text-blue-600 px-10 py-5 rounded-full font-bold text-lg hover:shadow-2xl hover:scale-105 transition-all duration-300">
            Get Started
          </button>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 1s ease-out;
        }
        
        .animate-slideUp {
          animation: slideUp 1s ease-out;
        }
      `}</style>
    </div>
  );
}