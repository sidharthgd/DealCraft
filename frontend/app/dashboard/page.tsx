'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { SearchBar } from '@/components/search/SearchBar'
import { UploadModal } from '@/components/upload/UploadModal'
import { useDealStore } from '@/store/dealStore'
import { useAuth } from '@/components/providers/AuthProvider'
import { Plus, FileText, Calendar, Loader2, Trash2, LogOut } from 'lucide-react'
import Link from 'next/link'

export default function DashboardPage() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const { deals, fetchDeals, deleteDeal } = useDealStore()
  const { logout, user } = useAuth()

  // Fetch deals when user is available and page loads
  useEffect(() => {
    // Only fetch deals if the auth state has resolved and the user is logged in
    if (!user) return

    const loadDeals = async () => {
      setIsLoading(true)
      await fetchDeals()
      setIsLoading(false)
    }

    loadDeals()
  }, [fetchDeals, user])

  const handleDeleteDeal = async (dealId: string, dealName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${dealName}"? This action cannot be undone and will delete all associated documents and memos.`)) {
      return
    }

    setIsDeleting(true)
    try {
      await deleteDeal(dealId)
    } catch (error) {
      console.error('Error deleting deal:', error)
      alert('Failed to delete deal. Please try again.')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">DealCraft AI</h1>
            <p className="text-gray-600 mt-2">AI-powered deal document analysis</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span>Welcome, {user?.displayName || user?.email}</span>
            </div>
            <Button 
              onClick={() => setIsUploadModalOpen(true)}
              className="flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              New Deal
            </Button>
            <Button 
              onClick={logout}
              variant="outline"
              className="flex items-center gap-2"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>

        <div className="mb-8">
          <SearchBar onSearch={(query) => console.log('Global search:', query)} />
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p className="text-gray-600">Loading deals...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {deals.map((deal) => (
                <Card key={deal.id} className="hover:shadow-md transition-shadow cursor-pointer relative">
                  <Link href={`/deals/${deal.id}`}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FileText className="h-5 w-5" />
                        {deal.name}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                          <Calendar className="h-4 w-4" />
                          {new Date(deal.created_at).toLocaleDateString()}
                        </div>
                        <div className="text-sm text-gray-600">
                          {deal.document_count} documents
                        </div>
                        <div className="text-xs text-gray-500 line-clamp-2">
                          {deal.description || 'No description available'}
                        </div>
                      </div>
                    </CardContent>
                  </Link>
                  
                  {/* Delete Button */}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2 h-8 w-8 text-gray-400 hover:text-red-600 hover:bg-red-50"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      handleDeleteDeal(deal.id, deal.name)
                    }}
                    disabled={isDeleting}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </Card>
              ))}
            </div>

            {deals.length === 0 && (
              <div className="text-center py-12">
                <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No deals yet</h3>
                <p className="text-gray-600 mb-6">Get started by creating your first deal</p>
                <Button onClick={() => setIsUploadModalOpen(true)}>
                  Create Deal
                </Button>
              </div>
            )}
          </>
        )}

        <UploadModal 
          isOpen={isUploadModalOpen}
          onClose={() => setIsUploadModalOpen(false)}
        />
      </div>
    </div>
  )
} 