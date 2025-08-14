'use client'

import { motion } from 'framer-motion'
import { ArrowLeft, Star, Clock, ChefHat, Leaf } from 'lucide-react'
import { useState } from 'react'
import Link from 'next/link'
import AIOrderChat from '../../components/AIOrderChat'

export default function MenuPage() {
  const [showChat, setShowChat] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState('all')

  const categories = [
    { id: 'all', name: 'All Items', icon: '🍽️' },
    { id: 'sandwiches', name: 'Sandwiches', icon: '🥪' },
    { id: 'salads', name: 'Salads', icon: '🥗' },
    { id: 'wraps', name: 'Wraps', icon: '🌯' },
    { id: 'sides', name: 'Sides', icon: '🍟' },
    { id: 'beverages', name: 'Beverages', icon: '🥤' }
  ]

  const menuItems = [
    // Sandwiches
    {
      name: "The Brooklyn Rose",
      description: "Premium pastrami, swiss cheese, coleslaw, Russian dressing on fresh rye bread",
      price: "$11.99",
      image: "/images/brooklyn-rose.jpg",
      category: "sandwiches",
      popular: true,
      badges: ["Signature", "Popular"]
    },
    {
      name: "Zu Zu Special",
      description: "Triple-stack turkey, ham, roast beef with avocado on sourdough bread",
      price: "$12.49",
      image: "/images/zuzu-special.jpg",
      category: "sandwiches",
      popular: false,
      badges: ["Triple Stack"]
    },
    {
      name: "The Big Lucky",
      description: "Monster sandwich with 5 meats (pastrami, turkey, ham, roast beef, salami), 2 cheeses, and avocado",
      price: "$13.99",
      image: "/images/monster-sandwich.jpg",
      category: "sandwiches",
      popular: true,
      badges: ["Monster Size", "5 Meats"]
    },
    {
      name: "Philly Cheesesteak",
      description: "Grilled ribeye steak, peppers, onions, provolone cheese on hoagie roll",
      price: "$11.99",
      image: "/images/philly-steak.jpg",
      category: "sandwiches",
      popular: true,
      badges: ["Grilled Fresh"]
    },
    {
      name: "Italian Supreme",
      description: "Salami, ham, pepperoni, provolone, lettuce, tomato, Italian dressing",
      price: "$10.99",
      image: "/images/zuzu-special.jpg",
      category: "sandwiches",
      popular: false,
      badges: ["Classic Italian"]
    },
    {
      name: "Turkey Club Deluxe",
      description: "Roasted turkey breast, bacon, avocado, lettuce, tomato on toasted sourdough",
      price: "$9.99",
      image: "/images/monster-sandwich.jpg",
      category: "sandwiches",
      popular: false,
      badges: ["Fresh Turkey"]
    },

    // Salads
    {
      name: "Honey BBQ Chicken Salad",
      description: "Grilled chicken with BBQ glaze, mixed greens, corn, tomatoes, cheddar cheese",
      price: "$9.99",
      image: "/images/chicken-salad.jpg",
      category: "salads",
      popular: true,
      badges: ["Protein Packed", "Gluten Free"]
    },
    {
      name: "Caesar Salad Supreme",
      description: "Crisp romaine lettuce, parmesan cheese, croutons, Caesar dressing",
      price: "$8.99",
      image: "/images/fresh-salads.jpg",
      category: "salads",
      popular: false,
      badges: ["Classic", "Fresh Daily"]
    },
    {
      name: "Cobb Salad Deluxe",
      description: "Mixed greens, grilled chicken, bacon, blue cheese, eggs, avocado, tomatoes",
      price: "$11.99",
      image: "/images/fresh-salads.jpg",
      category: "salads",
      popular: true,
      badges: ["Premium", "Complete Meal"]
    },
    {
      name: "Mediterranean Salad",
      description: "Mixed greens, olives, feta cheese, cucumbers, tomatoes, red onion, balsamic",
      price: "$9.49",
      image: "/images/fresh-salads.jpg",
      category: "salads",
      popular: false,
      badges: ["Vegetarian", "Mediterranean"]
    },

    // Wraps
    {
      name: "No Carb Classic Club",
      description: "Turkey, bacon, lettuce, tomato wrapped in fresh butter lettuce leaves",
      price: "$9.99",
      image: "/images/fresh-salads.jpg",
      category: "wraps",
      popular: false,
      badges: ["Keto Friendly", "Low Carb"]
    },
    {
      name: "Buffalo Chicken Wrap",
      description: "Spicy buffalo chicken, lettuce, tomato, ranch dressing in flour tortilla",
      price: "$8.99",
      image: "/images/chicken-salad.jpg",
      category: "wraps",
      popular: true,
      badges: ["Spicy", "Fan Favorite"]
    },
    {
      name: "Veggie Hummus Wrap",
      description: "Hummus, cucumbers, tomatoes, lettuce, sprouts, avocado in spinach tortilla",
      price: "$7.99",
      image: "/images/fresh-salads.jpg",
      category: "wraps",
      popular: false,
      badges: ["Vegan", "Healthy"]
    },

    // Sides
    {
      name: "Crispy Fries",
      description: "Golden crispy french fries seasoned with sea salt",
      price: "$3.99",
      image: "/images/deli-prep.jpg",
      category: "sides",
      popular: true,
      badges: ["Crispy", "Made Fresh"]
    },
    {
      name: "Onion Rings",
      description: "Beer-battered onion rings served with ranch dipping sauce",
      price: "$4.49",
      image: "/images/deli-prep.jpg",
      category: "sides",
      popular: false,
      badges: ["Beer Battered"]
    },
    {
      name: "Coleslaw",
      description: "Fresh cabbage slaw with creamy dressing and herbs",
      price: "$2.99",
      image: "/images/fresh-salads.jpg",
      category: "sides",
      popular: false,
      badges: ["Fresh Daily"]
    },
    {
      name: "Potato Salad",
      description: "Homestyle potato salad with herbs and mayo dressing",
      price: "$3.49",
      image: "/images/deli-prep.jpg",
      category: "sides",
      popular: true,
      badges: ["Homestyle"]
    },

    // Beverages
    {
      name: "Fountain Drinks",
      description: "Coke, Pepsi, Sprite, Orange, Root Beer, Diet options available",
      price: "$2.49",
      image: "/images/restaurant-ambiance.jpg",
      category: "beverages",
      popular: true,
      badges: ["Refillable"]
    },
    {
      name: "Fresh Iced Tea",
      description: "House-brewed iced tea, sweetened or unsweetened",
      price: "$2.99",
      image: "/images/restaurant-ambiance.jpg",
      category: "beverages",
      popular: false,
      badges: ["House Made"]
    },
    {
      name: "Coffee",
      description: "Freshly brewed coffee, regular or decaf",
      price: "$1.99",
      image: "/images/restaurant-ambiance.jpg",
      category: "beverages",
      popular: false,
      badges: ["Fresh Brewed"]
    }
  ]

  const filteredItems = selectedCategory === 'all' 
    ? menuItems 
    : menuItems.filter(item => item.category === selectedCategory)

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link 
                href="/"
                className="flex items-center space-x-2 text-gray-700 hover:text-red-600 transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Home</span>
              </Link>
              <div className="h-6 w-px bg-gray-300" />
              <div className="flex items-center space-x-3">
                <img src="/images/md-logo.png" alt="Mike&apos;s Deli Logo" className="h-10 w-auto" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">Mike&apos;s Deli</h1>
                  <p className="text-sm text-red-600">Complete Menu</p>
                </div>
              </div>
            </div>

            <motion.button
              onClick={() => setShowChat(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-red-600 text-white px-6 py-3 rounded-xl font-semibold shadow-lg hover:bg-red-700 hover:shadow-xl transition-all"
            >
              🤖 Order with AI
            </motion.button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 bg-gradient-to-r from-red-600 to-red-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
              Complete Menu
            </h1>
            <p className="text-xl text-red-100 mb-8">
              Over 15 years of deli excellence • Fresh ingredients daily
            </p>
            <div className="flex items-center justify-center space-x-8 text-red-100">
              <div className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>15-20 min pickup</span>
              </div>
              <div className="flex items-center space-x-2">
                <Star className="h-5 w-5 fill-current text-yellow-400" />
                <span>4.6/5 rating</span>
              </div>
              <div className="flex items-center space-x-2">
                <ChefHat className="h-5 w-5" />
                <span>Award winning</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Category Filter */}
      <section className="py-8 bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap justify-center gap-4">
            {categories.map((category) => (
              <motion.button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`px-6 py-3 rounded-xl font-semibold transition-all shadow-lg ${
                  selectedCategory === category.id
                    ? 'bg-red-600 text-white shadow-red-500/25'
                    : 'bg-white text-gray-700 hover:bg-red-50 hover:text-red-600 border-2 border-gray-200 hover:border-red-200'
                }`}
              >
                <span className="mr-2">{category.icon}</span>
                {category.name}
              </motion.button>
            ))}
          </div>
        </div>
      </section>

      {/* Menu Items */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {filteredItems.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.02, y: -5 }}
                className="bg-white rounded-3xl shadow-xl overflow-hidden border-2 border-transparent hover:border-red-200 transition-all relative group"
              >
                {/* Badges */}
                <div className="absolute top-4 left-4 z-10 flex flex-wrap gap-2">
                  {item.badges.map((badge, badgeIndex) => (
                    <span
                      key={badgeIndex}
                      className="bg-red-600 text-white text-xs px-2 py-1 rounded-full font-medium shadow-lg"
                    >
                      {badge}
                    </span>
                  ))}
                  {item.popular && (
                    <span className="bg-yellow-500 text-white text-xs px-2 py-1 rounded-full font-medium shadow-lg">
                      ⭐ Popular
                    </span>
                  )}
                </div>

                {/* Image */}
                <div className="aspect-w-16 aspect-h-12 overflow-hidden">
                  <img
                    src={item.image}
                    alt={item.name}
                    className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>

                {/* Content */}
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{item.name}</h3>
                  <p className="text-gray-600 mb-4 text-sm leading-relaxed">{item.description}</p>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-2xl font-bold text-red-600">{item.price}</span>
                    <motion.button
                      onClick={() => setShowChat(true)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="bg-red-600 text-white px-6 py-2 rounded-xl font-semibold hover:bg-red-700 hover:shadow-lg transition-all transform hover:scale-105"
                    >
                      Add to Order
                    </motion.button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-red-600 to-red-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl font-bold text-white mb-4">Ready to Order?</h2>
            <p className="text-xl text-red-100 mb-8">
              Use our AI chat assistant for instant ordering or call us directly
            </p>
            
            <div className="flex flex-wrap justify-center gap-4">
              <motion.button
                onClick={() => setShowChat(true)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-white text-red-600 px-8 py-4 rounded-xl text-lg font-semibold shadow-xl hover:shadow-2xl transition-all"
              >
                🤖 Order with AI Chat
              </motion.button>
              
              <motion.a
                href="tel:323-298-5960"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-red-800 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-xl hover:bg-red-900 hover:shadow-2xl transition-all"
              >
                📞 Call Slauson: (323) 298-5960
              </motion.a>
              
              <motion.a
                href="tel:213-617-8443"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-red-800 text-white px-8 py-4 rounded-xl text-lg font-semibold shadow-xl hover:bg-red-900 hover:shadow-2xl transition-all"
              >
                📞 Call Downtown: (213) 617-8443
              </motion.a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* AI Chat Component */}
      {showChat && (
        <AIOrderChat onClose={() => setShowChat(false)} />
      )}
    </div>
  )
}