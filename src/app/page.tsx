'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { Sandwich, Clock, MapPin, Phone, Star, ChefHat, Coffee } from 'lucide-react'
import { useState, useEffect } from 'react'
import AIOrderChat from '../components/AIOrderChat'

export default function Home() {
  const [showChat, setShowChat] = useState(false)
  const [currentSlide, setCurrentSlide] = useState(0)
  
  const heroImages = [
    '/images/deli-hero1.jpg',
    '/images/deli-hero2.jpg', 
    '/images/deli-hero3.jpg'
  ]
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % heroImages.length)
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-3"
            >
              <div className="flex items-center space-x-2">
                <img src="/images/md-logo.png" alt="Mike&apos;s Deli Logo" className="h-12 w-auto" />
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Mike&apos;s Deli</h1>
                  <p className="text-sm text-red-600">Fresh, Fast &amp; Delicious</p>
                </div>
              </div>
            </motion.div>
            
            <nav className="hidden md:flex space-x-8">
              <a href="/menu" className="text-gray-700 hover:text-red-600 font-medium">Full Menu</a>
              <a href="#locations" className="text-gray-700 hover:text-red-600 font-medium">Locations</a>
              <a href="#catering" className="text-gray-700 hover:text-red-600 font-medium">Catering</a>
              <div className="flex space-x-4">
                <a href="/dashboard" className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 hover:shadow-lg transition-all">
                  Admin Dashboard
                </a>
                <a href="/voice-demo" className="bg-white text-red-600 border-2 border-red-600 px-4 py-2 rounded-lg hover:bg-red-50 transition-all">
                  Voice AI Demo
                </a>
              </div>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Carousel Background */}
        <div className="absolute inset-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentSlide}
              initial={{ opacity: 0, scale: 1.1 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
              className="absolute inset-0"
            >
              <img
                src={heroImages[currentSlide]}
                alt={`Deli hero image ${currentSlide + 1}`}
                className="w-full h-full object-cover"
              />
            </motion.div>
          </AnimatePresence>
          <div className="absolute inset-0 bg-gradient-to-r from-black/70 via-black/50 to-black/60" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent" />
        </div>
        
        {/* Carousel Indicators */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex space-x-3 z-20">
          {heroImages.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentSlide(index)}
              className={`w-3 h-3 rounded-full transition-all duration-300 ${
                currentSlide === index 
                  ? 'bg-white shadow-lg scale-125' 
                  : 'bg-white/50 hover:bg-white/75'
              }`}
            />
          ))}
        </div>
        <div className="relative max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 drop-shadow-2xl">
                LA&apos;s Premier
                <span className="text-red-400"> Deli Experience</span>
              </h1>
              <p className="text-xl text-white/90 mb-8 leading-relaxed drop-shadow-lg">
                Over 15 years of crafting monster sandwiches and fresh salads. 
                Now powered by AI for instant ordering and 24/7 customer service.
              </p>
              
              <div className="flex flex-wrap gap-4">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowChat(true)}
                  className="bg-red-600 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-red-700 hover:shadow-red-500/25 transition-all"
                >
                  🤖 Order with AI Chat
                </motion.button>
                <motion.a
                  href="tel:323-298-5960"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-white text-gray-900 px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:shadow-3xl border-2 border-white hover:bg-red-50 hover:border-red-300 transition-all backdrop-blur-sm"
                >
                  📞 Call & Order
                </motion.a>
              </div>

              <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Star className="h-5 w-5 text-yellow-400 fill-current" />
                  <span className="text-white/80">4.6/5 Google Reviews</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="h-5 w-5 text-red-600" />
                  <span className="text-white/80">15-20 min pickup</span>
                </div>
                <div className="flex items-center space-x-2">
                  <MapPin className="h-5 w-5 text-red-600" />
                  <span className="text-white/80">2 LA locations</span>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative"
            >
              {/* Main Featured Item */}
              <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-8 transform rotate-1 border-2 border-red-100 hover:bg-white hover:shadow-3xl transition-all duration-300">
                <div className="mb-4">
                  <img src="/images/deli-hero1.jpg" alt="Brooklyn Rose Sandwich" className="w-full h-48 object-cover rounded-2xl mb-4" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">⭐ The Brooklyn Rose</h3>
                <p className="text-gray-700 mb-4">Premium pastrami, swiss cheese, coleslaw, Russian dressing on fresh rye bread</p>
                <div className="flex justify-between items-center">
                  <span className="text-3xl font-bold text-red-600">$11.99</span>
                  <button 
                    onClick={() => setShowChat(true)}
                    className="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition-all shadow-lg"
                  >
                    Order Now
                  </button>
                </div>
              </div>
              
              {/* Secondary Featured Item */}
              <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-8 transform -rotate-1 mt-6 border-2 border-red-100 hover:bg-white hover:shadow-3xl transition-all duration-300">
                <div className="mb-4">
                  <img src="/images/chicken-salad.jpg" alt="Honey BBQ Chicken Salad" className="w-full h-48 object-cover rounded-2xl mb-4" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">🥗 Honey BBQ Chicken Salad</h3>
                <p className="text-gray-700 mb-4">Grilled chicken with BBQ glaze, mixed greens, corn, tomatoes, cheddar</p>
                <div className="flex justify-between items-center">
                  <span className="text-3xl font-bold text-red-600">$9.99</span>
                  <button 
                    onClick={() => setShowChat(true)}
                    className="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition-all shadow-lg"
                  >
                    Order Now
                  </button>
                </div>
              </div>

              {/* Award Badge */}
              <div className="absolute -top-4 -right-4 bg-gradient-to-br from-red-600 to-red-700 text-white rounded-full p-4 shadow-2xl transform rotate-12 border-2 border-white">
                <div className="text-center">
                  <div className="text-lg font-bold">LA Weekly</div>
                  <div className="text-xs">Best Deli 2023</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Why Choose Mike&apos;s Deli?</h2>
            <p className="text-xl text-gray-600">Technology meets tradition for the perfect deli experience</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center p-6 rounded-xl bg-gray-50 border-2 border-gray-100"
            >
              <div className="bg-red-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <ChefHat className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Fresh Daily</h3>
              <p className="text-gray-600">All ingredients prepared fresh every morning using premium quality meats and produce</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-center p-6 rounded-xl bg-gray-50 border-2 border-gray-100"
            >
              <div className="bg-red-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Clock className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Lightning Fast</h3>
              <p className="text-gray-600">AI-powered ordering and kitchen optimization ensures 15-20 minute pickup times</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-center p-6 rounded-xl bg-gray-50 border-2 border-gray-100"
            >
              <div className="bg-red-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Phone className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">24/7 Voice AI</h3>
              <p className="text-gray-600">Call anytime and our AI voice agent will take your order, answer questions, and provide info</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="text-center p-6 rounded-xl bg-gray-50 border-2 border-gray-100"
            >
              <div className="bg-red-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Star className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Award Winning</h3>
              <p className="text-gray-600">LA Weekly &quot;Best Deli Sandwich&quot; 2023 and 4.6/5 Google rating from 1,200+ reviews</p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Menu Preview */}
      <section id="menu" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Signature Menu</h2>
            <p className="text-xl text-gray-600">Our most popular items loved by LA</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                name: "The Brooklyn Rose",
                description: "Premium pastrami, swiss, coleslaw, Russian dressing on rye",
                price: "$11.99",
                image: "/images/deli-hero1.jpg",
                popular: true
              },
              {
                name: "Zu Zu Special", 
                description: "Triple-stack turkey, ham, roast beef with avocado on sourdough",
                price: "$12.49",
                image: "/images/zuzu-special.jpg",
                popular: false
              },
              {
                name: "Honey BBQ Chicken Salad",
                description: "Grilled chicken, BBQ glaze, mixed greens, corn, cheddar",
                price: "$9.99", 
                image: "/images/chicken-salad.jpg",
                popular: true
              },
              {
                name: "The Big Lucky",
                description: "Monster sandwich with 5 meats, 2 cheeses, and avocado",
                price: "$13.99",
                image: "/images/monster-sandwich.jpg",
                popular: false
              },
              {
                name: "No Carb Classic Club",
                description: "Turkey, bacon, lettuce, tomato wrapped in butter lettuce",
                price: "$9.99",
                image: "/images/fresh-salads.jpg",
                popular: false
              },
              {
                name: "Philly Cheesesteak",
                description: "Grilled ribeye, peppers, onions, provolone on hoagie",
                price: "$11.99", 
                image: "/images/philly-steak.jpg",
                popular: true
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.02, y: -5 }}
                className="bg-white rounded-2xl shadow-xl p-0 cursor-pointer border-2 border-transparent hover:border-red-200 transition-all relative overflow-hidden group"
              >
                {item.popular && (
                  <div className="absolute top-3 right-3 bg-red-600 text-white text-xs px-2 py-1 rounded-full z-10 font-medium">
                    Popular
                  </div>
                )}
                
                <div className="aspect-w-16 aspect-h-12 mb-4 overflow-hidden rounded-t-2xl">
                  <img 
                    src={item.image} 
                    alt={item.name} 
                    className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300" 
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{item.name}</h3>
                  <p className="text-gray-600 mb-4 text-sm leading-relaxed">{item.description}</p>
                  <div className="flex justify-between items-center">
                    <span className="text-2xl font-bold text-red-600">{item.price}</span>
                    <button 
                      onClick={() => setShowChat(true)}
                      className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 hover:shadow-lg transition-all transform hover:scale-105"
                    >
                      Order Now
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="text-center mt-12 flex flex-wrap justify-center gap-4">
            <motion.a
              href="/menu"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-red-600 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-red-700 hover:shadow-red-500/25 transition-all inline-block"
            >
              View Full Menu
            </motion.a>
            <motion.button
              onClick={() => setShowChat(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-white text-red-600 border-2 border-red-600 px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-red-50 hover:shadow-red-500/25 transition-all"
            >
              🤖 Quick Order AI
            </motion.button>
          </div>
        </div>
      </section>

      {/* Gallery Section */}
      <section className="py-20 bg-gradient-to-br from-red-50 via-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Fresh Daily at Mike&apos;s</h2>
            <p className="text-xl text-gray-600">Behind the scenes of LA&apos;s premier deli experience</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/deli-prep.jpg" 
                alt="Fresh preparation at Mike's Deli" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">🥪 Fresh Daily Prep</h3>
                <p className="text-sm">Premium meats and ingredients prepared fresh every morning</p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/restaurant-ambiance.jpg" 
                alt="Mike's Deli interior atmosphere" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">🏪 Welcoming Atmosphere</h3>
                <p className="text-sm">Comfortable dining spaces perfect for any occasion</p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/fresh-salads.jpg" 
                alt="Fresh salads and healthy options" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">🥗 Healthy Options</h3>
                <p className="text-sm">Fresh salads and nutritious choices for every lifestyle</p>
              </div>
            </motion.div>
          </div>

          <div className="text-center mt-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="bg-white rounded-3xl shadow-2xl p-8 border-2 border-red-100 max-w-2xl mx-auto"
            >
              <div className="flex items-center justify-center space-x-4 mb-6">
                <div className="bg-red-600 p-3 rounded-full">
                  <ChefHat className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">Award Winning Quality</h3>
                  <p className="text-red-600 font-semibold">LA Weekly Best Deli 2023</p>
                </div>
              </div>
              <p className="text-gray-600 leading-relaxed">
                From our signature Brooklyn Rose to our monster Big Lucky sandwich, every item is crafted 
                with premium ingredients and over 15 years of deli expertise. Taste the difference quality makes.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Locations */}
      <section id="locations" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Our Locations</h2>
            <p className="text-xl text-gray-600">Serving Los Angeles with two convenient locations and exceptional atmosphere</p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-white rounded-3xl shadow-2xl overflow-hidden border-2 border-red-100 hover:shadow-3xl transition-all duration-300"
            >
              <div className="relative">
                <img 
                  src="/images/deli-interior.jpg" 
                  alt="Mike's Deli Interior - Slauson Location" 
                  className="w-full h-64 object-cover"
                />
                <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                  Main Location
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <MapPin className="h-6 w-6 text-red-600 mr-2" />
                  Slauson Avenue
                </h3>
                
                <div className="space-y-4 mb-8">
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-1">Address</p>
                    <p className="text-gray-700">4859 W. Slauson Avenue, Los Angeles, CA 90056</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-1">Phone</p>
                    <p className="text-gray-700">323-298-5960</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-2">Hours</p>
                    <div className="grid grid-cols-2 gap-2 text-sm text-gray-700">
                      <div>Mon-Fri: 8AM-8PM</div>
                      <div>Saturday: 8AM-8PM</div>
                      <div>Sunday: 10AM-5:30PM</div>
                      <div className="text-red-600 font-medium">Open Daily!</div>
                    </div>
                  </div>
                  
                  <div className="bg-red-50 rounded-xl p-4 border border-red-100">
                    <p className="font-semibold text-gray-900 mb-2">Features</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🅿️ Free Parking</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🚗 Drive-Thru</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🏪 Dine-In</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">📞 Catering</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <motion.a
                    href="tel:323-298-5960"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-red-600 text-white px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-red-700 hover:shadow-lg transition-all"
                  >
                    📞 Call Now
                  </motion.a>
                  <motion.button
                    onClick={() => setShowChat(true)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-white text-red-600 border-2 border-red-600 px-6 py-3 rounded-xl font-semibold flex-1 hover:bg-red-50 transition-all"
                  >
                    🤖 Order AI
                  </motion.button>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-white rounded-3xl shadow-2xl overflow-hidden border-2 border-red-100 hover:shadow-3xl transition-all duration-300"
            >
              <div className="relative">
                <img 
                  src="/images/restaurant-ambiance.jpg" 
                  alt="Mike's Deli Interior - Downtown Location" 
                  className="w-full h-64 object-cover"
                />
                <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                  Downtown
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <MapPin className="h-6 w-6 text-red-600 mr-2" />
                  Downtown LA
                </h3>
                
                <div className="space-y-4 mb-8">
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-1">Address</p>
                    <p className="text-gray-700">238 E. 1st Street, Los Angeles, CA 90012</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-1">Phone</p>
                    <p className="text-gray-700">213-617-8443</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="font-semibold text-gray-900 mb-2">Hours</p>
                    <div className="grid grid-cols-2 gap-2 text-sm text-gray-700">
                      <div>Mon-Fri: 8AM-8PM</div>
                      <div>Saturday: 8AM-8PM</div>
                      <div>Sunday: 10AM-5:30PM</div>
                      <div className="text-red-600 font-medium">Open Daily!</div>
                    </div>
                  </div>
                  
                  <div className="bg-red-50 rounded-xl p-4 border border-red-100">
                    <p className="font-semibold text-gray-900 mb-2">Features</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🚇 Metro Access</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🚶 Walk-In</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🏢 Business District</span>
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">🥪 Quick Service</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <motion.a
                    href="tel:213-617-8443"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-red-600 text-white px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-red-700 hover:shadow-lg transition-all"
                  >
                    📞 Call Now
                  </motion.a>
                  <motion.button
                    onClick={() => setShowChat(true)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-white text-red-600 border-2 border-red-600 px-6 py-3 rounded-xl font-semibold flex-1 hover:bg-red-50 transition-all"
                  >
                    🤖 Order AI
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </div>
          
          {/* Additional Location Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-16 bg-gradient-to-r from-red-50 to-white rounded-3xl p-8 border-2 border-red-100"
          >
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">🚗 Easy Ordering Options</h3>
              <p className="text-gray-600">Choose the most convenient way to get your Mike&apos;s Deli favorites</p>
            </div>
            
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg">
                <div className="bg-red-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Phone className="h-6 w-6 text-red-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Call Ahead</h4>
                <p className="text-sm text-gray-600">Skip the wait with phone orders</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg">
                <div className="bg-red-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Coffee className="h-6 w-6 text-red-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">AI Chat Order</h4>
                <p className="text-sm text-gray-600">24/7 intelligent ordering</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg">
                <div className="bg-red-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Sandwich className="h-6 w-6 text-red-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Walk-In</h4>
                <p className="text-sm text-gray-600">Fresh made while you wait</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg">
                <div className="bg-red-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <ChefHat className="h-6 w-6 text-red-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Catering</h4>
                <p className="text-sm text-gray-600">Large orders & events</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* AI Chat Component */}
      {showChat && (
        <AIOrderChat 
          onClose={() => setShowChat(false)}
        />
      )}

      {/* Floating AI Chat Button */}
      {!showChat && (
        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowChat(true)}
          className="fixed bottom-6 right-6 bg-red-600 text-white p-4 rounded-full shadow-2xl hover:bg-red-700 hover:shadow-red-500/25 transition-all z-50"
        >
          <Coffee className="h-6 w-6" />
        </motion.button>
      )}

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <img src="/images/md-logo.png" alt="Mike's Deli Logo" className="h-8 w-auto" />
                <h3 className="text-xl font-bold">Mike&apos;s Deli</h3>
              </div>
              <p className="text-gray-400">Fresh, Fast &amp; Delicious since 2009. Serving Los Angeles with premium deli classics and innovative AI-powered ordering.</p>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#menu" className="hover:text-red-400 transition-colors">Menu</a></li>
                <li><a href="#locations" className="hover:text-red-400 transition-colors">Locations</a></li>
                <li><a href="#catering" className="hover:text-red-400 transition-colors">Catering</a></li>
                <li><a href="/dashboard" className="hover:text-red-400 transition-colors">Admin Dashboard</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Contact</h4>
              <ul className="space-y-2 text-gray-400">
                <li>📞 Slauson: 323-298-5960</li>
                <li>📞 Downtown: 213-617-8443</li>
                <li>📧 mike@mikesdelionline.com</li>
                <li>🌐 mikesdelionline.com</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 Mike&apos;s Deli. All rights reserved. Powered by AI technology.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
