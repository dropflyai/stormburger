'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { Clock, MapPin, Phone, Star, ChefHat, Coffee, Zap, Flame } from 'lucide-react'
import { useState, useEffect } from 'react'
import AIOrderChat from '../components/AIOrderChat'

// Featured Rolodex Component
const FeaturedRolodex = ({ showChat, setShowChat }: { showChat: boolean; setShowChat: (show: boolean) => void }) => {
  const [currentCard, setCurrentCard] = useState(0)
  
  const featuredItems = [
    {
      name: "⚡ Classic StormBurger",
      description: "1/4 lb all-natural smashburger, cheese, lettuce, tomato, raw onions, pickles, storm sauce",
      price: "$8.99",
      image: "/images/storm-classic-burger.jpg",
      category: "Signature Burger"
    },
    {
      name: "🔥 Spicy Chicken Sandwich", 
      description: "All-natural handbreaded fried chicken, pickles, spicy mayo, thunder sauce",
      price: "$9.99",
      image: "/images/spicy-chicken.jpg",
      category: "Handbreaded Chicken"
    },
    {
      name: "🍟 Storm Fries",
      description: "Crispy golden french fries seasoned with our signature storm seasoning",
      price: "$4.99",
      image: "/images/storm-fries.jpg", 
      category: "Fresh Sides"
    },
    {
      name: "🧅 Thunder Onion Rings",
      description: "Handcrafted in-house with fresh onions, crispy golden perfection",
      price: "$5.99",
      image: "/images/onion-rings.jpg",
      category: "House Special"
    },
    {
      name: "🥛 Lightning Milkshake",
      description: "Thick and creamy vanilla milkshake topped with whipped cream",
      price: "$4.99", 
      image: "/images/milkshake.jpg",
      category: "Sweet Treats"
    },
    {
      name: "🍗 Chicken Strip Combo",
      description: "Hand-breaded chicken tenders with your choice of sauce",
      price: "$10.99",
      image: "/images/chicken-strips.jpg",
      category: "Crispy Strips"
    }
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentCard((prev) => (prev + 1) % featuredItems.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const getCardTransform = (index) => {
    const diff = index - currentCard
    const totalCards = featuredItems.length
    
    // Normalize the difference to handle wrapping
    let normalizedDiff = diff
    if (diff > totalCards / 2) normalizedDiff = diff - totalCards
    if (diff < -totalCards / 2) normalizedDiff = diff + totalCards
    
    const angle = normalizedDiff * 15 // Degrees between cards
    const distance = Math.abs(normalizedDiff) * 40 // Distance from center
    const scale = normalizedDiff === 0 ? 1 : 0.85 - Math.abs(normalizedDiff) * 0.1
    const opacity = normalizedDiff === 0 ? 1 : Math.max(0.3, 1 - Math.abs(normalizedDiff) * 0.2)
    const zIndex = 10 - Math.abs(normalizedDiff)
    
    return {
      transform: `perspective(1000px) rotateY(${angle}deg) translateZ(${distance}px) scale(${scale})`,
      opacity,
      zIndex
    }
  }

  return (
    <div className="relative h-96 w-full flex items-center justify-center overflow-visible">
      {/* Cards Container */}
      <div className="relative w-80 h-80">
        {featuredItems.map((item, index) => (
          <motion.div
            key={index}
            className="absolute inset-0 cursor-pointer"
            style={getCardTransform(index)}
            onClick={() => setCurrentCard(index)}
            whileHover={{ scale: index === currentCard ? 1.02 : 0.87 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
          >
            <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-6 border-2 border-blue-100 hover:bg-white hover:shadow-3xl transition-all duration-300 h-full">
              <div className="mb-4 relative overflow-hidden rounded-2xl">
                <img 
                  src={item.image} 
                  alt={item.name} 
                  className="w-full h-36 object-cover transform hover:scale-105 transition-transform duration-300" 
                />
                <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-medium">
                  {item.category}
                </div>
              </div>
              
              <h3 className="text-lg font-bold text-gray-900 mb-2 line-clamp-1">{item.name}</h3>
              <p className="text-gray-700 mb-3 text-sm line-clamp-2">{item.description}</p>
              
              <div className="flex justify-between items-center">
                <span className="text-2xl font-bold text-blue-600">{item.price}</span>
                <button 
                  onClick={(e) => {
                    e.stopPropagation()
                    setShowChat(true)
                  }}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-all shadow-lg text-sm font-semibold"
                >
                  Order Now
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Navigation Dots */}
      <div className="absolute -bottom-16 left-1/2 transform -translate-x-1/2 flex space-x-2">
        {featuredItems.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentCard(index)}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              currentCard === index 
                ? 'bg-blue-600 shadow-lg scale-125' 
                : 'bg-blue-300 hover:bg-blue-400'
            }`}
          />
        ))}
      </div>

      {/* Award Badge */}
      <div className="absolute -top-4 -right-4 bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-full p-4 shadow-2xl transform rotate-12 border-2 border-white z-20">
        <div className="text-center">
          <div className="text-lg font-bold">Est. 2023</div>
          <div className="text-xs">Fresh Daily</div>
        </div>
      </div>
    </div>
  )
}

export default function Home() {
  const [showChat, setShowChat] = useState(false)
  const [currentSlide, setCurrentSlide] = useState(0)
  
  const heroImages = [
    '/images/storm-hero1.jpg',
    '/images/storm-hero2.jpg', 
    '/images/burger-grill.jpg'
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
      <header className="bg-white shadow-lg border-b-2 border-blue-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-3"
            >
              <div className="flex items-center space-x-2">
                <img src="/images/storm-burger-logo.jpg" alt="Storm Burger Logo" className="h-12 w-auto" />
                <div>
                  <h1 className="text-2xl font-bold text-blue-900">StormBurger</h1>
                  <p className="text-sm text-blue-600">Fresh • Fast • Electric</p>
                </div>
              </div>
            </motion.div>
            
            <nav className="hidden md:flex space-x-8">
              <a href="/menu" className="text-gray-700 hover:text-blue-600 font-medium transition-colors">Full Menu</a>
              <a href="#locations" className="text-gray-700 hover:text-blue-600 font-medium transition-colors">Locations</a>
              <a href="#catering" className="text-gray-700 hover:text-blue-600 font-medium transition-colors">Food Truck</a>
              <div className="flex space-x-4">
                <a href="/dashboard" className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 hover:shadow-lg transition-all">
                  Admin Dashboard
                </a>
                <a href="/voice-demo" className="bg-white text-blue-600 border-2 border-blue-600 px-4 py-2 rounded-lg hover:bg-blue-50 transition-all">
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
                alt={`Storm Burger hero image ${currentSlide + 1}`}
                className="w-full h-full object-cover"
              />
            </motion.div>
          </AnimatePresence>
          <div className="absolute inset-0 bg-gradient-to-r from-blue-900/80 via-blue-800/60 to-blue-900/70" />
          <div className="absolute inset-0 bg-gradient-to-t from-blue-900/50 via-transparent to-transparent" />
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
                LA&apos;s Electric
                <span className="text-blue-300"> Burger Experience</span>
              </h1>
              <p className="text-xl text-white/90 mb-8 leading-relaxed drop-shadow-lg">
                Fresh, all-natural smashburgers and handbreaded chicken delivered daily. 
                Now powered by AI for lightning-fast ordering and 24/7 customer service.
              </p>
              
              <div className="flex flex-wrap gap-4">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowChat(true)}
                  className="bg-blue-600 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-blue-700 hover:shadow-blue-500/25 transition-all"
                >
                  ⚡ Order with AI Chat
                </motion.button>
                <motion.a
                  href="tel:323-298-5960"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-white text-blue-900 px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:shadow-3xl border-2 border-white hover:bg-blue-50 hover:border-blue-300 transition-all backdrop-blur-sm"
                >
                  📞 Call & Order
                </motion.a>
              </div>

              <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Star className="h-5 w-5 text-yellow-400 fill-current" />
                  <span className="text-white/80">4.8/5 Yelp Reviews</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="h-5 w-5 text-blue-300" />
                  <span className="text-white/80">Under 12 min wait</span>
                </div>
                <div className="flex items-center space-x-2">
                  <MapPin className="h-5 w-5 text-blue-300" />
                  <span className="text-white/80">2 LA locations</span>
                </div>
              </div>
            </motion.div>

            {/* 3D Rolodex Featured Items */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative h-96 w-full"
            >
              <FeaturedRolodex showChat={showChat} setShowChat={setShowChat} />
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
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Why Choose StormBurger?</h2>
            <p className="text-xl text-gray-600">Technology meets tradition for the perfect burger experience</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center p-6 rounded-xl bg-blue-50 border-2 border-blue-100"
            >
              <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <ChefHat className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Fresh Daily</h3>
              <p className="text-gray-600">All-natural meat and fresh bread delivered daily by local Inglewood businesses</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-center p-6 rounded-xl bg-blue-50 border-2 border-blue-100"
            >
              <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Lightning Fast</h3>
              <p className="text-gray-600">AI-powered ordering and optimized kitchen ensures under 12-minute wait times</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-center p-6 rounded-xl bg-blue-50 border-2 border-blue-100"
            >
              <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Phone className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">24/7 Voice AI</h3>
              <p className="text-gray-600">Call anytime and our AI voice agent will take your order and answer questions</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="text-center p-6 rounded-xl bg-blue-50 border-2 border-blue-100"
            >
              <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Star className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Community Focused</h3>
              <p className="text-gray-600">Founded by Chef John Herndon to serve quality food to overlooked communities</p>
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
                name: "Classic StormBurger",
                description: "1/4 lb all-natural smashburger, cheese, lettuce, tomato, raw onions, pickles, storm sauce",
                price: "$8.99",
                image: "/images/storm-classic-burger.jpg",
                popular: true
              },
              {
                name: "Bacon BBQ Burger", 
                description: "1/4 lb all-natural smashburger with cheese, bacon, onion straws, BBQ sauce",
                price: "$10.99",
                image: "/images/double-classic-burger.jpg",
                popular: false
              },
              {
                name: "Double Classic",
                description: "Two 1/4 lb all-natural smashburgers, cheese, lettuce, tomato, raw onions, pickles, burger sauce",
                price: "$12.99", 
                image: "/images/double-classic-burger.jpg",
                popular: true
              },
              {
                name: "Jalapeño Lightning",
                description: "Spicy burger with jalapeños, pepper jack cheese, and thunder sauce",
                price: "$9.99",
                image: "/images/jalapeno-lightning.jpg",
                popular: false
              },
              {
                name: "Classic Chicken Sandwich",
                description: "All-natural handbreaded fried chicken, pickles, mayo on fresh bun",
                price: "$8.99",
                image: "/images/chicken-sandwich.jpg",
                popular: false
              },
              {
                name: "Spicy Chicken Sandwich",
                description: "All-natural handbreaded fried chicken, pickles, spicy mayo, thunder sauce",
                price: "$9.99", 
                image: "/images/spicy-chicken.jpg",
                popular: true
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.02, y: -5 }}
                className="bg-white rounded-2xl shadow-xl p-0 cursor-pointer border-2 border-transparent hover:border-blue-200 transition-all relative overflow-hidden group"
              >
                {item.popular && (
                  <div className="absolute top-3 right-3 bg-blue-600 text-white text-xs px-2 py-1 rounded-full z-10 font-medium">
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
                    <span className="text-2xl font-bold text-blue-600">{item.price}</span>
                    <button 
                      onClick={() => setShowChat(true)}
                      className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 hover:shadow-lg transition-all transform hover:scale-105"
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
              className="bg-blue-600 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-blue-700 hover:shadow-blue-500/25 transition-all inline-block"
            >
              View Full Menu
            </motion.a>
            <motion.button
              onClick={() => setShowChat(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-white text-blue-600 border-2 border-blue-600 px-8 py-4 rounded-xl text-lg font-semibold shadow-2xl hover:bg-blue-50 hover:shadow-blue-500/25 transition-all"
            >
              ⚡ Quick Order AI
            </motion.button>
          </div>
        </div>
      </section>

      {/* Gallery Section */}
      <section className="py-20 bg-gradient-to-br from-blue-50 via-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Fresh Daily at StormBurger</h2>
            <p className="text-xl text-gray-600">Behind the scenes of LA&apos;s electric burger experience</p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/burger-grill.jpg" 
                alt="Fresh preparation at StormBurger" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-blue-900/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">⚡ Fresh Grilled Daily</h3>
                <p className="text-sm">All-natural meat grilled fresh to order on our signature smash technique</p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/burger-restaurant.jpg" 
                alt="StormBurger service atmosphere" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-blue-900/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">🏃 Fast Service</h3>
                <p className="text-sm">Drive-thru and walk-up service with under 12-minute wait times</p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative overflow-hidden rounded-3xl shadow-2xl group"
            >
              <img 
                src="/images/chicken-sandwich.jpg" 
                alt="Handbreaded chicken preparation" 
                className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-blue-900/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute bottom-6 left-6 right-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <h3 className="text-xl font-bold mb-2">🍗 Handbreaded Chicken</h3>
                <p className="text-sm">All-natural chicken handbreaded and fried fresh to order</p>
              </div>
            </motion.div>
          </div>

          <div className="text-center mt-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="bg-white rounded-3xl shadow-2xl p-8 border-2 border-blue-100 max-w-2xl mx-auto"
            >
              <div className="flex items-center justify-center space-x-4 mb-6">
                <div className="bg-blue-600 p-3 rounded-full">
                  <Flame className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">Electric Quality</h3>
                  <p className="text-blue-600 font-semibold">Founded by Former Umami Burger Chef</p>
                </div>
              </div>
              <p className="text-gray-600 leading-relaxed">
                From our signature Classic StormBurger to our spicy Thunder Sauce, every item is crafted 
                with local Inglewood suppliers and over 15 years of burger expertise. Taste the storm.
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
            <p className="text-xl text-gray-600">Serving LA with fresh burgers and expanding across the city</p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-white rounded-3xl shadow-2xl overflow-hidden border-2 border-blue-100 hover:shadow-3xl transition-all duration-300"
            >
              <div className="relative">
                <img 
                  src="/images/burger-restaurant.jpg" 
                  alt="StormBurger Inglewood Location" 
                  className="w-full h-64 object-cover"
                />
                <div className="absolute top-4 left-4 bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                  Main Location
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <MapPin className="h-6 w-6 text-blue-600 mr-2" />
                  Inglewood (La Brea)
                </h3>
                
                <div className="space-y-4 mb-8">
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-1">Address</p>
                    <p className="text-gray-700">1500 N La Brea Ave, Inglewood, CA 90302</p>
                  </div>
                  
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-2">Hours</p>
                    <div className="text-sm text-gray-700">
                      <div className="font-medium text-blue-600 mb-1">7:00 AM - 11:00 PM Daily</div>
                      <div>Breakfast: 7:00 AM - 11:00 AM</div>
                      <div>Lunch/Dinner: 11:00 AM - 11:00 PM</div>
                    </div>
                  </div>
                  
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-2">Features</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">🚗 Drive-Thru</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">🚶 Walk-Up</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">🥞 Breakfast</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">🚚 Food Truck</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <motion.button
                    onClick={() => setShowChat(true)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-blue-700 hover:shadow-lg transition-all"
                  >
                    ⚡ Order AI
                  </motion.button>
                  <motion.a
                    href="mailto:events@stormburger.com"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-white text-blue-600 border-2 border-blue-600 px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-blue-50 transition-all"
                  >
                    📧 Catering
                  </motion.a>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-white rounded-3xl shadow-2xl overflow-hidden border-2 border-blue-100 hover:shadow-3xl transition-all duration-300"
            >
              <div className="relative">
                <img 
                  src="/images/burger-grill.jpg" 
                  alt="StormBurger Long Beach Coming Soon" 
                  className="w-full h-64 object-cover"
                />
                <div className="absolute top-4 left-4 bg-yellow-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                  Coming Soon
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <MapPin className="h-6 w-6 text-blue-600 mr-2" />
                  Long Beach
                </h3>
                
                <div className="space-y-4 mb-8">
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-1">Address</p>
                    <p className="text-gray-700">5801 Cherry Ave, Long Beach, CA</p>
                    <p className="text-sm text-blue-600 mt-1">Former Church&apos;s location</p>
                  </div>
                  
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-2">Opening Soon</p>
                    <div className="text-sm text-gray-700">
                      <div className="font-medium text-yellow-600 mb-1">Food truck available daily</div>
                      <div>Full restaurant opening 2024</div>
                    </div>
                  </div>
                  
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <p className="font-semibold text-gray-900 mb-2">Future Expansion</p>
                    <div className="flex flex-wrap gap-2">
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">🏗️ Under Development</span>
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">🚚 Food Truck Active</span>
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">📍 Compton Next</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <motion.a
                    href="mailto:events@stormburger.com"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-blue-700 hover:shadow-lg transition-all"
                  >
                    📧 Events & Catering
                  </motion.a>
                  <motion.a
                    href="https://www.instagram.com/stormburger/"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-white text-blue-600 border-2 border-blue-600 px-6 py-3 rounded-xl font-semibold flex-1 text-center hover:bg-blue-50 transition-all"
                  >
                    📱 Follow Updates
                  </motion.a>
                </div>
              </div>
            </motion.div>
          </div>
          
          {/* Additional Location Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-16 bg-gradient-to-r from-blue-50 to-white rounded-3xl p-8 border-2 border-blue-100"
          >
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">⚡ Electric Ordering Options</h3>
              <p className="text-gray-600">Choose the most convenient way to get your StormBurger favorites</p>
            </div>
            
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg border border-blue-100">
                <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Phone className="h-6 w-6 text-blue-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Call Ahead</h4>
                <p className="text-sm text-gray-600">Skip the wait with phone orders</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg border border-blue-100">
                <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Coffee className="h-6 w-6 text-blue-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">AI Chat Order</h4>
                <p className="text-sm text-gray-600">24/7 intelligent ordering</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg border border-blue-100">
                <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Zap className="h-6 w-6 text-blue-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Drive-Thru</h4>
                <p className="text-sm text-gray-600">Fresh made while you wait</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-2xl shadow-lg border border-blue-100">
                <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <ChefHat className="h-6 w-6 text-blue-600" />
                </div>
                <h4 className="font-bold text-gray-900 mb-2">Food Truck</h4>
                <p className="text-sm text-gray-600">Events & catering available</p>
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
          className="fixed bottom-6 right-6 bg-blue-600 text-white p-4 rounded-full shadow-2xl hover:bg-blue-700 hover:shadow-blue-500/25 transition-all z-50"
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
                <img src="/images/storm-favicon.png" alt="Storm Burger Logo" className="h-8 w-auto" />
                <h3 className="text-xl font-bold">StormBurger</h3>
              </div>
              <p className="text-gray-400">Fresh • Fast • Electric since 2023. Serving LA with premium burgers and innovative AI-powered ordering from our Inglewood community.</p>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="/menu" className="hover:text-blue-400 transition-colors">Full Menu</a></li>
                <li><a href="#locations" className="hover:text-blue-400 transition-colors">Locations</a></li>
                <li><a href="mailto:events@stormburger.com" className="hover:text-blue-400 transition-colors">Food Truck Events</a></li>
                <li><a href="/dashboard" className="hover:text-blue-400 transition-colors">Admin Dashboard</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Contact</h4>
              <ul className="space-y-2 text-gray-400">
                <li>📍 1500 N La Brea Ave, Inglewood, CA</li>
                <li>🕐 7:00 AM - 11:00 PM Daily</li>
                <li>📧 events@stormburger.com</li>
                <li>📱 @stormburger on Instagram</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 StormBurger. All rights reserved. Powered by AI technology.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}