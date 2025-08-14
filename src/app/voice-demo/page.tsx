'use client'

import { motion } from 'framer-motion'
import { useState, useRef, useEffect } from 'react'
import { ArrowLeft, Phone, Mic, MicOff, Volume2, VolumeX, Play, Pause } from 'lucide-react'
import Link from 'next/link'

interface CallLog {
  id: string
  timestamp: string
  duration: string
  customerName: string
  phone: string
  orderTotal: number
  status: 'completed' | 'in_progress' | 'failed'
  transcript: string
}

const mockCallLogs: CallLog[] = [
  {
    id: '001',
    timestamp: '2:34 PM',
    duration: '3:42',
    customerName: 'David Park',
    phone: '(323) 555-0123',
    orderTotal: 24.98,
    status: 'completed',
    transcript: 'AI: Hi! Thanks for calling Mike\'s Deli. I\'m your AI assistant. How can I help you today?\nCustomer: Hi, I\'d like to place an order for pickup.\nAI: Great! Which location would you prefer - Slauson or Downtown?\nCustomer: Slauson please.\nAI: Perfect. What can I get started for you?\nCustomer: I\'ll take one Brooklyn Rose and one Honey BBQ Chicken Salad.\nAI: Excellent choices! That\'s our signature Brooklyn Rose sandwich and our popular Honey BBQ Chicken Salad. Your total is $21.98. May I have your name and phone number?\nCustomer: David Park, 323-555-0123.\nAI: Thank you David. Your order will be ready for pickup at our Slauson location in about 15 minutes. You\'ll receive a confirmation text shortly. Is there anything else?\nCustomer: That\'s it, thanks!\nAI: Perfect! See you soon at Mike\'s Deli!'
  },
  {
    id: '002',
    timestamp: '1:28 PM',
    duration: '2:15',
    customerName: 'Maria Santos',
    phone: '(213) 555-0456',
    orderTotal: 13.99,
    status: 'completed',
    transcript: 'AI: Hello! Mike\'s Deli AI assistant here. How may I help you?\nCustomer: What are your hours?\nAI: We\'re open Monday through Friday 8 AM to 8 PM, Saturday 8 AM to 8 PM, and Sunday 10 AM to 5:30 PM. Both our Slauson and Downtown locations have the same hours.\nCustomer: Great, and can I order now?\nAI: Absolutely! What would you like today?\nCustomer: I\'ll take the Big Lucky sandwich.\nAI: The Big Lucky - great choice! That\'s our monster sandwich with 5 meats, 2 cheeses, and avocado for $13.99. Which location for pickup?\nCustomer: Downtown please.\nAI: Perfect. Your name and phone number?\nCustomer: Maria Santos, 213-555-0456.\nAI: Thank you Maria! Your Big Lucky will be ready at our Downtown location in 18 minutes. You\'ll get a text confirmation.'
  }
]

export default function VoiceDemo() {
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [currentCall, setCurrentCall] = useState<CallLog | null>(null)
  const [isPlayingDemo, setIsPlayingDemo] = useState(false)
  const [demoStep, setDemoStep] = useState(0)
  const [showTranscript, setShowTranscript] = useState(false)

  const demoConversation = [
    { speaker: 'AI', text: 'Hi! Thanks for calling Mike\'s Deli. I\'m your AI assistant. How can I help you today?' },
    { speaker: 'Customer', text: 'Hi, I\'d like to order a sandwich for pickup.' },
    { speaker: 'AI', text: 'Great! Which location would you prefer - our Slauson location or Downtown LA?' },
    { speaker: 'Customer', text: 'Slauson please.' },
    { speaker: 'AI', text: 'Perfect! What sandwich would you like today?' },
    { speaker: 'Customer', text: 'What do you recommend?' },
    { speaker: 'AI', text: 'Our most popular is The Brooklyn Rose - premium pastrami with swiss cheese, coleslaw, and Russian dressing on fresh rye bread for $11.99. Customers say it\'s amazing!' },
    { speaker: 'Customer', text: 'That sounds perfect, I\'ll take one.' },
    { speaker: 'AI', text: 'Excellent choice! That\'s one Brooklyn Rose for $11.99. Can I get your name and phone number for the order?' },
    { speaker: 'Customer', text: 'Sure, it\'s John Smith, 323-555-1234.' },
    { speaker: 'AI', text: 'Thank you John! Your Brooklyn Rose will be ready for pickup at our Slauson location in about 15 minutes. You\'ll receive a confirmation text with your order details. Is there anything else I can help you with?' },
    { speaker: 'Customer', text: 'No that\'s perfect, thank you!' },
    { speaker: 'AI', text: 'Wonderful! Thank you for choosing Mike\'s Deli. Have a great day!' }
  ]

  const startDemo = () => {
    setIsPlayingDemo(true)
    setDemoStep(0)
    setShowTranscript(true)
    playDemoStep(0)
  }

  const stopDemo = () => {
    setIsPlayingDemo(false)
    setDemoStep(0)
    setIsSpeaking(false)
  }

  const playDemoStep = (step: number) => {
    if (step >= demoConversation.length) {
      setIsPlayingDemo(false)
      setIsSpeaking(false)
      return
    }

    setDemoStep(step)
    setIsSpeaking(true)

    // Simulate speech duration based on text length
    const duration = Math.max(2000, demoConversation[step].text.length * 50)
    
    setTimeout(() => {
      setIsSpeaking(false)
      setTimeout(() => {
        playDemoStep(step + 1)
      }, 500)
    }, duration)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-red-50">
      {/* Header */}
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link 
                href="/"
                className="flex items-center space-x-2 text-gray-600 hover:text-orange-600 transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Site</span>
              </Link>
              <div className="h-6 w-px bg-gray-300" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Voice AI Demo</h1>
                <p className="text-sm text-gray-600">24/7 AI Phone Assistant</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                🎤 Voice AI Online
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Live Demo Section */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl shadow-xl p-8"
            >
              <div className="text-center mb-8">
                <div className="bg-gradient-to-r from-orange-500 to-red-600 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Phone className="h-12 w-12 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Call Mike&apos;s Deli</h2>
                <p className="text-gray-600">Experience our AI phone assistant</p>
                
                <div className="mt-6 space-y-2">
                  <div className="flex items-center justify-center space-x-6 text-lg font-semibold">
                    <span className="text-orange-600">📞 Slauson: (323) 298-5960</span>
                  </div>
                  <div className="flex items-center justify-center space-x-6 text-lg font-semibold">
                    <span className="text-orange-600">📞 Downtown: (213) 617-8443</span>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Try the Demo Conversation</h3>
                <p className="text-gray-600 mb-6">Listen to a sample interaction between our AI and a customer</p>
                
                <div className="flex justify-center mb-6">
                  {!isPlayingDemo ? (
                    <button
                      onClick={startDemo}
                      className="bg-gradient-to-r from-orange-500 to-red-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all flex items-center space-x-2"
                    >
                      <Play className="h-5 w-5" />
                      <span>Play Demo Call</span>
                    </button>
                  ) : (
                    <button
                      onClick={stopDemo}
                      className="bg-red-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg transition-all flex items-center space-x-2"
                    >
                      <Pause className="h-5 w-5" />
                      <span>Stop Demo</span>
                    </button>
                  )}
                </div>

                {/* Voice Animation */}
                {isSpeaking && (
                  <div className="flex justify-center mb-6">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse" />
                      <div className="w-3 h-6 bg-orange-400 rounded-full animate-pulse" style={{ animationDelay: '0.1s' }} />
                      <div className="w-3 h-4 bg-orange-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
                      <div className="w-3 h-8 bg-orange-600 rounded-full animate-pulse" style={{ animationDelay: '0.3s' }} />
                      <div className="w-3 h-5 bg-orange-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
                      <Volume2 className="h-6 w-6 text-orange-600 ml-2" />
                    </div>
                  </div>
                )}

                {/* Live Transcript */}
                {showTranscript && (
                  <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                    <h4 className="font-medium text-gray-900 mb-3">Live Transcript:</h4>
                    <div className="space-y-2">
                      {demoConversation.slice(0, demoStep + 1).map((line, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          className={`text-sm ${
                            line.speaker === 'AI' 
                              ? 'text-orange-700 font-medium' 
                              : 'text-gray-700'
                          } ${index === demoStep && isSpeaking ? 'bg-orange-100 p-2 rounded' : ''}`}
                        >
                          <span className="font-semibold">{line.speaker}:</span> {line.text}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Features */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-2xl shadow-lg p-6 mt-6"
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4">Voice AI Features</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg">
                  <Phone className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900">Natural Speech</h4>
                  <p className="text-xs text-gray-600">Human-like conversations</p>
                </div>
                
                <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg">
                  <Mic className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900">Smart Recognition</h4>
                  <p className="text-xs text-gray-600">Understands complex orders</p>
                </div>
                
                <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg">
                  <Volume2 className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900">24/7 Availability</h4>
                  <p className="text-xs text-gray-600">Never miss an order</p>
                </div>
                
                <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg">
                  <Play className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900">Instant Processing</h4>
                  <p className="text-xs text-gray-600">Real-time order placement</p>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Call Logs */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-2xl shadow-lg"
            >
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-bold text-gray-900">Recent Voice AI Calls</h2>
                <p className="text-sm text-gray-600">Automated phone orders processed today</p>
              </div>
              
              <div className="divide-y divide-gray-200">
                {mockCallLogs.map((call, index) => (
                  <motion.div
                    key={call.id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-6 hover:bg-gray-50 cursor-pointer"
                    onClick={() => setCurrentCall(call)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          call.status === 'completed' ? 'bg-green-100 text-green-600' :
                          call.status === 'in_progress' ? 'bg-blue-100 text-blue-600' :
                          'bg-red-100 text-red-600'
                        }`}>
                          <Phone className="h-5 w-5" />
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">{call.customerName}</h3>
                          <p className="text-sm text-gray-600">{call.phone}</p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">${call.orderTotal}</p>
                        <p className="text-xs text-gray-500">{call.timestamp} • {call.duration}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Call Transcript */}
            {currentCall && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl shadow-lg p-6 mt-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-900">Call Transcript</h3>
                  <button
                    onClick={() => setCurrentCall(null)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-900">Customer:</span> {currentCall.customerName}
                    </div>
                    <div>
                      <span className="font-medium text-gray-900">Phone:</span> {currentCall.phone}
                    </div>
                    <div>
                      <span className="font-medium text-gray-900">Duration:</span> {currentCall.duration}
                    </div>
                    <div>
                      <span className="font-medium text-gray-900">Total:</span> ${currentCall.orderTotal}
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                  <div className="whitespace-pre-line text-sm text-gray-700">
                    {currentCall.transcript}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-2xl shadow-lg p-6 mt-6"
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4">Voice AI Performance</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-600">98.5%</p>
                  <p className="text-sm text-gray-600">Success Rate</p>
                </div>
                
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">2.8 min</p>
                  <p className="text-sm text-gray-600">Avg Call Time</p>
                </div>
                
                <div className="text-center">
                  <p className="text-2xl font-bold text-orange-600">156</p>
                  <p className="text-sm text-gray-600">Calls Today</p>
                </div>
                
                <div className="text-center">
                  <p className="text-2xl font-bold text-purple-600">4.9/5</p>
                  <p className="text-sm text-gray-600">Customer Rating</p>
                </div>
              </div>
              
              <div className="mt-6 pt-6 border-t border-gray-200 text-center">
                <p className="text-sm text-gray-600 mb-2">AI handles calls automatically</p>
                <p className="text-xs text-gray-500">Escalates complex issues to staff when needed</p>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}