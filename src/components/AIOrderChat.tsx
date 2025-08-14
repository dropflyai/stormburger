'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useRef, useEffect } from 'react'
import { X, Send, ShoppingCart, Clock, Star, Phone } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface MenuItem {
  id: string
  name: string
  price: number
  description: string
  category: string
  emoji: string
}

interface CartItem extends MenuItem {
  quantity: number
  customizations?: string[]
}

interface AIOrderChatProps {
  onClose: () => void
}

export default function AIOrderChat({ onClose }: AIOrderChatProps) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm Mike's AI assistant. I can help you place an order, answer questions about our menu, or check store hours. What would you like today?",
      timestamp: Date.now()
    }
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [cart, setCart] = useState<CartItem[]>([])
  const [showCart, setShowCart] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const menuItems: MenuItem[] = [
    { id: '1', name: 'The Brooklyn Rose', price: 11.99, description: 'Premium pastrami, swiss, coleslaw, Russian dressing on rye', category: 'Sandwiches', emoji: '🥪' },
    { id: '2', name: 'Zu Zu Special', price: 12.49, description: 'Triple-stack turkey, ham, roast beef with avocado on sourdough', category: 'Sandwiches', emoji: '🥪' },
    { id: '3', name: 'The Mo Mo', price: 10.99, description: 'Italian combo with salami, mortadella, provolone, Italian dressing', category: 'Sandwiches', emoji: '🥪' },
    { id: '4', name: 'Mike Deli #1', price: 9.49, description: 'Roast beef with cheddar, horseradish mayo on French bread', category: 'Sandwiches', emoji: '🥪' },
    { id: '5', name: 'The Big Lucky', price: 13.99, description: 'Monster sandwich with 5 meats, 2 cheeses, and avocado', category: 'Sandwiches', emoji: '🥪' },
    { id: '6', name: 'Classic Club', price: 10.49, description: 'Turkey, bacon, lettuce, tomato, mayo on toasted white bread', category: 'Sandwiches', emoji: '🥪' },
    { id: '7', name: 'Philly Cheesesteak', price: 11.99, description: 'Grilled ribeye, peppers, onions, provolone on hoagie', category: 'Sandwiches', emoji: '🥪' },
    { id: '8', name: 'Honey BBQ Chicken Salad', price: 9.99, description: 'Grilled chicken, BBQ glaze, mixed greens, corn, cheddar', category: 'Salads', emoji: '🥗' },
    { id: '9', name: 'Chef Salad', price: 8.99, description: 'Ham, turkey, swiss, cheddar, tomatoes, cucumbers, hard-boiled egg', category: 'Salads', emoji: '🥗' },
    { id: '10', name: 'Grilled Chicken Caesar', price: 9.49, description: 'Romaine, grilled chicken, parmesan, croutons, Caesar dressing', category: 'Salads', emoji: '🥗' },
    { id: '11', name: 'No Carb Classic Club', price: 9.99, description: 'Turkey, bacon, tomato, mayo wrapped in butter lettuce', category: 'Lettuce Wraps', emoji: '🥬' },
    { id: '12', name: 'No Carb Big Lucky', price: 12.99, description: 'Turkey, ham, pastrami, cheese, avocado in lettuce wraps', category: 'Lettuce Wraps', emoji: '🥬' }
  ]

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const addToCart = (item: MenuItem, quantity: number = 1) => {
    const existingItem = cart.find(cartItem => cartItem.id === item.id)
    if (existingItem) {
      setCart(cart.map(cartItem => 
        cartItem.id === item.id 
          ? { ...cartItem, quantity: cartItem.quantity + quantity }
          : cartItem
      ))
    } else {
      setCart([...cart, { ...item, quantity }])
    }
    toast.success(`Added ${item.name} to cart!`)
  }

  const removeFromCart = (itemId: string) => {
    setCart(cart.filter(item => item.id !== itemId))
    toast.success('Item removed from cart')
  }

  const updateQuantity = (itemId: string, newQuantity: number) => {
    if (newQuantity === 0) {
      removeFromCart(itemId)
      return
    }
    setCart(cart.map(item => 
      item.id === itemId ? { ...item, quantity: newQuantity } : item
    ))
  }

  const getCartTotal = () => {
    return cart.reduce((total, item) => total + (item.price * item.quantity), 0)
  }

  const processMessage = async (userMessage: string) => {
    setIsTyping(true)
    
    // Simple AI response logic based on keywords
    const lowerMessage = userMessage.toLowerCase()
    let response = ''
    
    if (lowerMessage.includes('menu') || lowerMessage.includes('what do you have') || lowerMessage.includes('options')) {
      response = `Here are our popular menu categories:

🥪 **Signature Sandwiches:**
• The Brooklyn Rose - $11.99 (our best seller!)
• Zu Zu Special - $12.49 (triple-stack)  
• The Big Lucky - $13.99 (monster sandwich)
• Philly Cheesesteak - $11.99

🥗 **Fresh Salads:**
• Honey BBQ Chicken Salad - $9.99
• Chef Salad - $8.99
• Grilled Chicken Caesar - $9.49

🥬 **No-Carb Lettuce Wraps:**
• No Carb Classic Club - $9.99
• No Carb Big Lucky - $12.99

Would you like to add any of these to your order?`
    
    } else if (lowerMessage.includes('brooklyn rose') || lowerMessage.includes('pastrami')) {
      const item = menuItems.find(i => i.name === 'The Brooklyn Rose')
      if (item) {
        response = `Great choice! The Brooklyn Rose is our signature sandwich 🌟

${item.emoji} **${item.name}** - $${item.price}
${item.description}

This is our most popular sandwich - customers say it "puts Bay Cities' godmother to shame!" 

Would you like to add this to your cart? Just say "add it" or "add Brooklyn Rose"`
      }
    
    } else if (lowerMessage.includes('hours') || lowerMessage.includes('open') || lowerMessage.includes('time')) {
      response = `Our hours are:

📍 **Slauson Location (Main):**
Monday-Friday: 8:00 AM - 8:00 PM
Saturday: 8:00 AM - 8:00 PM  
Sunday: 10:00 AM - 5:30 PM
📞 323-298-5960

📍 **Downtown LA:**
Monday-Friday: 8:00 AM - 8:00 PM
Saturday: 8:00 AM - 8:00 PM
Sunday: 10:00 AM - 5:30 PM  
📞 213-617-8443

Both locations are open now! Orders typically ready in 15-20 minutes.`

    } else if (lowerMessage.includes('add') && (lowerMessage.includes('brooklyn') || lowerMessage.includes('rose'))) {
      const item = menuItems.find(i => i.name === 'The Brooklyn Rose')
      if (item) {
        addToCart(item)
        response = `Perfect! Added The Brooklyn Rose to your cart 🛒

Current cart:
${cart.length + 1} item(s) - $${(getCartTotal() + item.price).toFixed(2)}

Anything else you'd like to add? Or would you like to review your order?`
      }

    } else if (lowerMessage.includes('add') && lowerMessage.includes('zuzu')) {
      const item = menuItems.find(i => i.name === 'Zu Zu Special')
      if (item) {
        addToCart(item)
        response = `Added the Zu Zu Special to your cart! That's a hearty triple-stack sandwich 🥪

Current total: $${(getCartTotal() + item.price).toFixed(2)}

What else can I get for you today?`
      }

    } else if (lowerMessage.includes('cart') || lowerMessage.includes('order') || lowerMessage.includes('total')) {
      if (cart.length === 0) {
        response = `Your cart is currently empty 🛒

Would you like me to recommend some of our most popular items?`
      } else {
        const total = getCartTotal()
        response = `Here's your current order:

${cart.map(item => `${item.emoji} ${item.name} x${item.quantity} - $${(item.price * item.quantity).toFixed(2)}`).join('\n')}

**Total: $${total.toFixed(2)}**

Ready to checkout? I can help you complete your order for pickup!`
      }

    } else if (lowerMessage.includes('checkout') || lowerMessage.includes('place order') || lowerMessage.includes('complete')) {
      if (cart.length === 0) {
        response = `You don't have any items in your cart yet! Would you like to browse our menu?`
      } else {
        response = `Great! Let's complete your order 📝

**Order Summary:**
${cart.map(item => `• ${item.name} x${item.quantity}`).join('\n')}

**Total: $${getCartTotal().toFixed(2)}**

Which location would you like to pick up from?
1. Slauson (323-298-5960) - Ready in 15-20 min
2. Downtown (213-617-8443) - Ready in 15-20 min

I'll also need your name and phone number to complete the order!`
      }

    } else if (lowerMessage.includes('price') || lowerMessage.includes('cost') || lowerMessage.includes('$')) {
      response = `Here are our price ranges:

🥪 **Sandwiches:** $9.49 - $13.99
🥗 **Salads:** $8.99 - $9.99  
🥬 **Lettuce Wraps:** $9.99 - $12.99

Our most popular items:
• The Brooklyn Rose - $11.99
• Zu Zu Special - $12.49
• Honey BBQ Chicken Salad - $9.99

All portions are generous - our customers call them "monster" sandwiches! What sounds good to you?`

    } else if (lowerMessage.includes('delivery') || lowerMessage.includes('deliver')) {
      response = `We offer several delivery options:

🚚 **Third-Party Delivery:**
• DoorDash, Uber Eats, Grubhub available
• Typical delivery time: 30-45 minutes

📦 **Pickup (Recommended):**
• Ready in 15-20 minutes  
• No delivery fees
• Guaranteed fresh and hot

🏢 **Catering Delivery:**
• Available for orders over $100
• 24-hour advance notice preferred

Would you like to place an order for pickup or delivery?`

    } else if (lowerMessage.includes('phone') || lowerMessage.includes('call')) {
      response = `You can call us directly:

📞 **Slauson:** 323-298-5960
📞 **Downtown:** 213-617-8443

**NEW:** We also have 24/7 AI voice ordering! Call anytime and our voice assistant will take your order, answer questions, and process payment over the phone.

But I'm happy to help you order right here in chat too! What would you like?`

    } else {
      // Default friendly response
      response = `I'm here to help with your Mike's Deli order! 😊

I can help you:
• Browse our menu and add items to cart
• Check store hours and locations  
• Answer questions about ingredients
• Complete your order for pickup
• Provide nutritional information

What would you like to know about? Or are you ready to place an order?`
    }

    // Simulate typing delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000))
    
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: response,
      timestamp: Date.now()
    }])
    setIsTyping(false)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = {
      role: 'user' as const,
      content: input.trim(),
      timestamp: Date.now()
    }

    setMessages(prev => [...prev, userMessage])
    processMessage(input.trim())
    setInput('')
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl h-[600px] flex"
      >
        {/* Chat Section */}
        <div className="flex-1 flex flex-col">
          {/* Chat Header */}
          <div className="bg-red-600 text-white p-4 rounded-t-2xl flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                🤖
              </div>
              <div>
                <h3 className="font-semibold">Mike&apos;s AI Assistant</h3>
                <p className="text-sm opacity-90">Ready to take your order!</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-full transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user' 
                    ? 'bg-red-600 text-white' 
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  <p className="text-sm whitespace-pre-line">{message.content}</p>
                </div>
              </motion.div>
            ))}
            
            {isTyping && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-gray-100 text-gray-900 px-4 py-2 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input */}
          <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything or say 'menu' to browse..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              />
              <button
                type="submit"
                disabled={!input.trim()}
                className="bg-red-600 text-white p-2 rounded-lg hover:bg-red-700 hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </form>
        </div>

        {/* Cart Sidebar */}
        <div className="w-80 border-l border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-gray-900">Your Order</h4>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <ShoppingCart className="h-4 w-4" />
                <span>{cart.reduce((sum, item) => sum + item.quantity, 0)} items</span>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {cart.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <ShoppingCart className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Your cart is empty</p>
                <p className="text-sm">Ask me about our menu!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {cart.map((item) => (
                  <motion.div
                    key={item.id}
                    layout
                    className="bg-white border border-gray-200 rounded-lg p-3"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <h5 className="font-medium text-gray-900 text-sm">{item.name}</h5>
                        <p className="text-xs text-gray-600">${item.price.toFixed(2)} each</p>
                      </div>
                      <button
                        onClick={() => removeFromCart(item.id)}
                        className="text-red-500 hover:text-red-700 text-xs"
                      >
                        Remove
                      </button>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => updateQuantity(item.id, item.quantity - 1)}
                          className="w-6 h-6 bg-gray-200 rounded-full flex items-center justify-center text-sm hover:bg-gray-300"
                        >
                          -
                        </button>
                        <span className="text-sm font-medium w-8 text-center">{item.quantity}</span>
                        <button
                          onClick={() => updateQuantity(item.id, item.quantity + 1)}
                          className="w-6 h-6 bg-gray-200 rounded-full flex items-center justify-center text-sm hover:bg-gray-300"
                        >
                          +
                        </button>
                      </div>
                      <span className="font-semibold text-sm">${(item.price * item.quantity).toFixed(2)}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {cart.length > 0 && (
            <div className="border-t border-gray-200 p-4 bg-gray-50">
              <div className="flex justify-between items-center mb-3">
                <span className="font-semibold">Total:</span>
                <span className="font-bold text-lg text-red-600">${getCartTotal().toFixed(2)}</span>
              </div>
              <div className="space-y-2">
                <button
                  onClick={() => {
                    const message = `I'd like to checkout with my current order: ${cart.map(item => `${item.name} x${item.quantity}`).join(', ')}. Total: $${getCartTotal().toFixed(2)}`
                    setInput(message)
                  }}
                  className="w-full bg-red-600 text-white py-2 rounded-lg hover:bg-red-700 hover:shadow-lg transition-all font-medium"
                >
                  Complete Order
                </button>
                <div className="flex items-center justify-center space-x-4 text-xs text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>15-20 min</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Phone className="h-3 w-3" />
                    <span>Call to order</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}