import {
  Brain,
  Mail,
  Github,
  Linkedin,
  Twitter,
  ExternalLink,
  Heart,
} from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const quickLinks = [
    { name: "Home", href: "/" },
    { name: "TB Diagnosis", href: "/diagnosis/tb" },
    { name: "Animal Classifier", href: "/diagnosis/animal" },
    { name: "About Project", href: "/about" },
  ];

  const resources = [
    { name: "Documentation", href: "#" },
    { name: "Research Paper", href: "#" },
    { name: "API Reference", href: "#" },
    { name: "Privacy Policy", href: "#" },
  ];

  const socialLinks = [
    { name: "GitHub", icon: Github, href: "#", color: "hover:text-gray-900" },
    {
      name: "LinkedIn",
      icon: Linkedin,
      href: "#",
      color: "hover:text-blue-600",
    },
    { name: "Twitter", icon: Twitter, href: "#", color: "hover:text-blue-400" },
    {
      name: "Email",
      icon: Mail,
      href: "mailto:contact@xai-diagnosis.com",
      color: "hover:text-red-500",
    },
  ];

  const handleLinkClick = (href) => {
    // In a real app with React Router, use: navigate(href)
    console.log("Navigate to:", href);
  };

  return (
    <footer className="bg-gray-900 text-gray-300">
      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="bg-linear-to-br from-blue-500 to-indigo-600 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-white font-bold text-lg">XAI Diagnosis</h3>
                <p className="text-xs text-gray-400">Explainable AI</p>
              </div>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              Making AI predictions transparent and understandable through
              advanced explainability techniques.
            </p>
            <div className="flex space-x-3">
              {socialLinks.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.name}
                    href={social.href}
                    className={`text-gray-400 ${social.color} transition-colors duration-200`}
                    aria-label={social.name}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon className="w-5 h-5" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.name}>
                  <button
                    onClick={() => handleLinkClick(link.href)}
                    className="text-gray-400 hover:text-white transition-colors duration-200 text-sm flex items-center group"
                  >
                    <span className="group-hover:translate-x-1 transition-transform duration-200">
                      {link.name}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              {resources.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-gray-400 hover:text-white transition-colors duration-200 text-sm flex items-center group"
                  >
                    <span className="group-hover:translate-x-1 transition-transform duration-200">
                      {link.name}
                    </span>
                    <ExternalLink className="w-3 h-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact & Newsletter */}
          <div>
            <h4 className="text-white font-semibold mb-4">Stay Updated</h4>
            <p className="text-sm text-gray-400 mb-4">
              Get the latest updates on XAI research and features.
            </p>
            <div className="flex">
              <input
                type="email"
                placeholder="Enter your email"
                className="flex-1 px-4 py-2 rounded-l-lg bg-gray-800 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-r-lg transition-colors duration-200">
                <Mail className="w-4 h-4 text-white" />
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              We respect your privacy. Unsubscribe anytime.
            </p>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            {/* Copyright */}
            <div className="text-sm text-gray-400">
              © {currentYear} XAI Diagnosis System. All rights reserved.
            </div>

            {/* Tech Stack Badge */}
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <span>Built with</span>
              <Heart className="w-4 h-4 text-red-500 fill-current" />
              <span>using React, TensorFlow & FastAPI</span>
            </div>

            {/* Legal Links */}
            <div className="flex space-x-6 text-sm">
              <a
                href="#"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                Terms
              </a>
              <a
                href="#"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                Privacy
              </a>
              <a
                href="#"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                Cookies
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Decorative Gradient Line */}
      <div className="h-1 bg-linear-to-r from-blue-600 via-indigo-600 to-purple-600"></div>
    </footer>
  );
};

export default Footer;
