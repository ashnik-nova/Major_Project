import { useState } from "react";
import { Menu, X, Brain, Home, Upload, Info } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";

const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  const navigation = [
    { name: "Home", href: "/", icon: Home },
    { name: "Model Testing", href: "classification", icon: Upload },
    
  ];

  const handleNavClose = () => setIsOpen(false);

  return (
    <nav className="bg-white shadow-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo & Brand */}
          <div className="flex items-center">
            <button
              onClick={() => {
                navigate("/");
                handleNavClose();
              }}
              className="flex items-center space-x-2 group"
              aria-label="Go to home"
            >
              <div className="bg-linear-to-br from-blue-600 to-indigo-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div className="hidden sm:block">
                <span className="text-xl font-bold bg-linear-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  XAI Diagnosis
                </span>
                <p className="text-xs text-gray-500">Explainable AI</p>
              </div>
            </button>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  to={item.href}
                  key={item.name}
                  className={({ isActive }) =>
                    `flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition-colors duration-200 font-medium ${
                      isActive ? "bg-blue-50 text-blue-600 font-semibold" : ""
                    }`
                  }
                  onClick={handleNavClose}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </div>

          {/* CTA Button (Desktop) */}
          <div className="hidden md:block">
            <button
              onClick={() => {
                navigate("/classification");
                handleNavClose();
              }}
              className="bg-linear-to-r from-blue-600 to-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:shadow-lg hover:scale-105 transition-all duration-200"
            >
              Start 
            </button>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-700 hover:text-blue-600 focus:outline-none transition-colors"
              aria-label="Toggle menu"
            >
              {isOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`md:hidden overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="px-4 pt-2 pb-4 space-y-1 bg-gray-50 border-t border-gray-200">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                to={item.href}
                key={item.name}
                className={({ isActive }) =>
                  `w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition-colors duration-200 font-medium ${
                    isActive ? "bg-blue-50 text-blue-600 font-semibold" : ""
                  }`
                }
                onClick={handleNavClose}
              >
                <Icon className="w-5 h-5" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}

          {/* Mobile CTA */}
          <button
            onClick={() => {
              navigate("/diagnosis/tb");
              handleNavClose();
            }}
            className="w-full mt-2 bg-linear-to-r from-blue-600 to-indigo-600 text-white px-4 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-200"
          >
            Start Diagnosis
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Header;
