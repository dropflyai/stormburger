'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { 
  ArrowLeft, 
  Phone, 
  MessageSquare, 
  Clock, 
  DollarSign, 
  TrendingUp,
  Users,
  CheckCircle,
  AlertCircle,
  Eye,
  Star,
  Volume2
} from 'lucide-react'
import Link from 'next/link'

interface Order {
  id: string
  customerName: string
  phone: string
  items: Array<{
    name: string
    quantity: number
    price: number
  }>
  total: number
  status: 'pending' | 'preparing' | 'ready' | 'completed'
  orderTime: string
  pickupTime: string
  location: 'slauson' | 'downtown'
  source: 'ai_chat' | 'voice_ai' | 'phone' | 'walk_in'
}

const mockOrders: Order[] = [
  {
    id: '001',
    customerName: 'Sarah Chen',
    phone: '(323) 555-0123',
    items: [
      { name: 'The Brooklyn Rose', quantity: 1, price: 11.99 },
      { name: 'Honey BBQ Chicken Salad', quantity: 1, price: 9.99 }
    ],
    total: 21.98,
    status: 'preparing',
    orderTime: '12:15 PM',
    pickupTime: '12:35 PM',
    location: 'slauson',
    source: 'ai_chat'
  },
  {
    id: '002',
    customerName: 'Marcus Johnson',
    phone: '(213) 555-0456',
    items: [
      { name: 'Zu Zu Special', quantity: 2, price: 12.49 },
      { name: 'Classic Club', quantity: 1, price: 10.49 }
    ],
    total: 35.47,
    status: 'ready',
    orderTime: '12:05 PM',
    pickupTime: '12:25 PM',
    location: 'downtown',
    source: 'voice_ai'
  },
  {
    id: '003',
    customerName: 'Jennifer Rodriguez',
    phone: '(323) 555-0789',
    items: [
      { name: 'The Big Lucky', quantity: 1, price: 13.99 }
    ],
    total: 13.99,
    status: 'pending',
    orderTime: '12:18 PM',
    pickupTime: '12:38 PM',
    location: 'slauson',
    source: 'phone'
  }
]

export default function Dashboard() {
  const [orders, setOrders] = useState<Order[]>(mockOrders)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null)

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const updateOrderStatus = (orderId: string, newStatus: Order['status']) => {
    setOrders(orders.map(order => 
      order.id === orderId ? { ...order, status: newStatus } : order
    ))
  }

  const getStatusColor = (status: Order['status']) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800'
      case 'preparing': return 'bg-blue-100 text-blue-800'
      case 'ready': return 'bg-green-100 text-green-800'
      case 'completed': return 'bg-gray-100 text-gray-800'
    }
  }

  const getSourceIcon = (source: Order['source']) => {
    switch (source) {
      case 'ai_chat': return <MessageSquare className="h-4 w-4" />
      case 'voice_ai': return <Volume2 className="h-4 w-4" />
      case 'phone': return <Phone className="h-4 w-4" />
      case 'walk_in': return <Users className="h-4 w-4" />
    }
  }

  const getSourceLabel = (source: Order['source']) => {
    switch (source) {
      case 'ai_chat': return 'AI Chat'
      case 'voice_ai': return 'Voice AI'
      case 'phone': return 'Phone'
      case 'walk_in': return 'Walk-in'
    }
  }

  const todayStats = {
    totalOrders: 47,
    totalRevenue: 642.83,
    averageOrderValue: 13.67,
    aiOrders: 28,
    satisfaction: 4.8
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
                <h1 className="text-2xl font-bold text-gray-900">Mike&apos;s Deli Dashboard</h1>
                <p className="text-sm text-gray-600">{currentTime.toLocaleString()}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                ✅ Both Locations Online
              </div>
              <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                🤖 AI Systems Active
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <div className="flex items-center">
              <div className="bg-blue-500 p-3 rounded-lg">
                <Clock className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Today&apos;s Orders</p>
                <p className="text-2xl font-bold text-gray-900">{todayStats.totalOrders}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <div className="flex items-center">
              <div className="bg-green-500 p-3 rounded-lg">
                <DollarSign className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Revenue</p>
                <p className="text-2xl font-bold text-gray-900">${todayStats.totalRevenue}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <div className="flex items-center">
              <div className="bg-purple-500 p-3 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Avg Order</p>
                <p className="text-2xl font-bold text-gray-900">${todayStats.averageOrderValue}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <div className="flex items-center">
              <div className="bg-orange-500 p-3 rounded-lg">
                <MessageSquare className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">AI Orders</p>
                <p className="text-2xl font-bold text-gray-900">{todayStats.aiOrders}</p>
                <p className="text-xs text-orange-600">{Math.round(todayStats.aiOrders/todayStats.totalOrders*100)}% of total</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <div className="flex items-center">
              <div className="bg-yellow-500 p-3 rounded-lg">
                <Star className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Satisfaction</p>
                <p className="text-2xl font-bold text-gray-900">{todayStats.satisfaction}</p>
                <p className="text-xs text-yellow-600">⭐⭐⭐⭐⭐</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Recent Orders */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl shadow-lg"
            >
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-bold text-gray-900">Live Orders</h2>
                <p className="text-sm text-gray-600">Real-time order tracking across both locations</p>
              </div>
              
              <div className="divide-y divide-gray-200">
                {orders.map((order, index) => (
                  <motion.div
                    key={order.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-6 hover:bg-gray-50 cursor-pointer"
                    onClick={() => setSelectedOrder(order)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {getSourceIcon(order.source)}
                          <span className="text-xs text-gray-500">{getSourceLabel(order.source)}</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">#{order.id} - {order.customerName}</h3>
                          <p className="text-sm text-gray-600">{order.items.length} items • ${order.total}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <p className="text-sm text-gray-900">{order.pickupTime}</p>
                          <p className="text-xs text-gray-500">{order.location}</p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(order.status)}`}>
                          {order.status}
                        </span>
                        <Eye className="h-4 w-4 text-gray-400" />
                      </div>
                    </div>
                    
                    {order.status !== 'completed' && (
                      <div className="mt-4 flex space-x-2">
                        {order.status === 'pending' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              updateOrderStatus(order.id, 'preparing')
                            }}
                            className="bg-blue-500 text-white px-3 py-1 rounded-lg text-sm hover:bg-blue-600 transition-colors"
                          >
                            Start Preparing
                          </button>
                        )}
                        {order.status === 'preparing' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              updateOrderStatus(order.id, 'ready')
                            }}
                            className="bg-green-500 text-white px-3 py-1 rounded-lg text-sm hover:bg-green-600 transition-colors"
                          >
                            Mark Ready
                          </button>
                        )}
                        {order.status === 'ready' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              updateOrderStatus(order.id, 'completed')
                            }}
                            className="bg-gray-500 text-white px-3 py-1 rounded-lg text-sm hover:bg-gray-600 transition-colors"
                          >
                            Complete
                          </button>
                        )}
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Order Details Sidebar */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              {selectedOrder ? (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-gray-900">Order #{selectedOrder.id}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedOrder.status)}`}>
                      {selectedOrder.status}
                    </span>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-900">Customer</h4>
                      <p className="text-sm text-gray-600">{selectedOrder.customerName}</p>
                      <p className="text-sm text-gray-600">{selectedOrder.phone}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">Order Details</h4>
                      <div className="space-y-2">
                        {selectedOrder.items.map((item, index) => (
                          <div key={index} className="flex justify-between text-sm">
                            <span>{item.name} x{item.quantity}</span>
                            <span>${(item.price * item.quantity).toFixed(2)}</span>
                          </div>
                        ))}
                      </div>
                      <div className="border-t pt-2 mt-2">
                        <div className="flex justify-between font-medium">
                          <span>Total</span>
                          <span>${selectedOrder.total.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">Timing</h4>
                      <p className="text-sm text-gray-600">Ordered: {selectedOrder.orderTime}</p>
                      <p className="text-sm text-gray-600">Pickup: {selectedOrder.pickupTime}</p>
                      <p className="text-sm text-gray-600">Location: {selectedOrder.location}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900">Source</h4>
                      <div className="flex items-center space-x-2">
                        {getSourceIcon(selectedOrder.source)}
                        <span className="text-sm text-gray-600">{getSourceLabel(selectedOrder.source)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <Eye className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Select an order to view details</p>
                </div>
              )}
            </motion.div>

            {/* AI System Status */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-xl shadow-lg p-6 mt-6"
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4">AI System Status</h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <MessageSquare className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-700">Chat AI</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs text-green-600">Online</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Volume2 className="h-4 w-4 text-orange-500" />
                    <span className="text-sm text-gray-700">Voice AI</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs text-green-600">Online</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-purple-500" />
                    <span className="text-sm text-gray-700">Order Processing</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs text-green-600">Optimal</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-700">Kitchen Display</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs text-green-600">Connected</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="text-center">
                  <p className="text-sm text-gray-600">AI Processed Orders Today</p>
                  <p className="text-2xl font-bold text-orange-600">{todayStats.aiOrders}</p>
                  <p className="text-xs text-gray-500">59.6% of total orders</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}