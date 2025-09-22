'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SearchBar } from '@/components/search/SearchBar'
import { SimpleSearchInput } from '@/components/search/SimpleSearchInput'
import { ResultsPane } from '@/components/search/ResultsPane'
import dynamic from 'next/dynamic'
import { MemoEditor } from '@/components/memo/MemoEditor'
import { UploadModal } from '@/components/upload/UploadModal'
import { useDealStore } from '@/store/dealStore'
import { useDocuments } from '@/hooks/useDocuments'
import { useSearch } from '@/hooks/useSearch'
import { Button } from '@/components/ui/button'
import { Plus, FileText, Search, BookOpen, Loader2, ArrowLeft } from 'lucide-react'
import { useMemo } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Deal } from '@/types'
import Link from 'next/link'

// IMPORTANT: Load the PDF viewer only on the client to avoid DOM-specific
// APIs (e.g., DOMMatrix) during server rendering in Cloud Run.
const DocViewer = dynamic(
  () => import('@/components/document/DocViewer').then(m => m.DocViewer),
  {
    ssr: false,
    loading: () => (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading viewer…</p>
      </div>
    ),
  }
)

export default function DealPage() {
  const params = useParams()
  const dealId = params.id as string
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({})
  const [activeTab, setActiveTab] = useState('memos')
  const [deal, setDeal] = useState<Deal | null>(null)
  const [isDealLoading, setIsDealLoading] = useState(true)
  const [dealNotFound, setDealNotFound] = useState(false)
  
  const { deals, fetchDeal } = useDealStore()
  const { documents } = useDocuments(dealId)
  const { searchResults, answer, search, isSearching } = useSearch()
  
  const selectedDocument = documents.find(d => d.id === selectedDocumentId)

  // List of categories (should match backend)
  const CATEGORIES = [
    'Income Statement',
    'Balance Sheet',
    'Cash Flow',
    'LOI',
    'CIM',
    'Diligence Tracker',
    'Customer List',
    'General Document', // Keep for uncategorized
  ]

  // Group documents by category - moved this hook before any early returns
  const docsByCategory = useMemo(() => {
    const grouped: Record<string, typeof documents> = {}
    
    // Initialize all known categories
    CATEGORIES.forEach(cat => { grouped[cat] = [] })
    
    // Add documents to their categories
    documents.forEach(doc => {
      const cat = doc.category || 'General Document'
      if (!grouped[cat]) {
        grouped[cat] = [] // Handle any unexpected categories
      }
      grouped[cat].push(doc)
    })
    
    return grouped
  }, [documents])

  // Fetch deal data
  useEffect(() => {
    const loadDeal = async () => {
      setIsDealLoading(true)
      setDealNotFound(false)
      
      // First check if deal is already in store
      const existingDeal = deals.find(d => d.id === dealId)
      if (existingDeal) {
        setDeal(existingDeal)
        setIsDealLoading(false)
        return
      }

      // If not in store, fetch from backend
      try {
        const fetchedDeal = await fetchDeal(dealId)
        if (fetchedDeal) {
          setDeal(fetchedDeal)
        } else {
          setDealNotFound(true)
        }
      } catch (error) {
        console.error('Error loading deal:', error)
        setDealNotFound(true)
      } finally {
        setIsDealLoading(false)
      }
    }

    if (dealId) {
      loadDeal()
    }
  }, [dealId, deals, fetchDeal])

  // Function to handle search within this deal
  const handleDealSearch = (query: string) => {
    search(query, dealId)
    // Switch to search results tab when search is performed
    setActiveTab('search-results')
  }

  // Loading state - now using conditional rendering instead of early return
  if (isDealLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading deal...</p>
        </div>
      </div>
    )
  }

  // Deal not found state - now using conditional rendering instead of early return
  if (dealNotFound || !deal) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <FileText className="h-16 w-16 mx-auto mb-4 text-gray-300" />
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">Deal Not Found</h1>
          <p className="text-gray-600 mb-4">
            The deal you're looking for doesn't exist or you don't have access to it.
          </p>
          <Link href="/dashboard">
            <Button>
              Go Back to Dashboard
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Sidebar - Documents */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0" title="Back to Dashboard">
                <ArrowLeft className="h-4 w-4" />
              </Button>
            </Link>
            <h2 className="text-lg font-semibold flex-1">{deal.name}</h2>
            <Button 
              size="sm" 
              onClick={() => setIsUploadModalOpen(true)}
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-sm text-gray-600">{documents.length} documents</p>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          {CATEGORIES.map(category => (
            <div key={category}>
              <button
                className="w-full flex items-center justify-between px-4 py-2 text-left hover:bg-gray-50 focus:outline-none border-b border-gray-100 bg-gray-50"
                onClick={() => setOpenCategories(prev => ({ ...prev, [category]: !prev[category] }))}
              >
                <span className="font-medium text-gray-800">{category}</span>
                {openCategories[category] ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </button>
              {openCategories[category] && docsByCategory[category] && docsByCategory[category].length > 0 && (
                <div className="pl-2">
                  {docsByCategory[category].map((doc) => (
                    <div
                      key={doc.id}
                      className={`p-3 border-b border-gray-100 cursor-pointer hover:bg-gray-50 ${selectedDocumentId === doc.id ? 'bg-blue-50 border-blue-200' : ''}`}
                      onClick={() => setSelectedDocumentId(doc.id)}
                    >
                      <div className="flex items-start gap-3">
                        <FileText className="h-5 w-5 text-gray-400 mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <h3 className="text-sm font-medium text-gray-900 truncate">{doc.name}</h3>
                          <p className="text-xs text-gray-500 mt-1">{new Date(doc.created_at).toLocaleDateString()}</p>
                          <p className="text-xs text-gray-500">{doc.file_size ? `${Math.round(doc.file_size / 1024)} KB` : ''}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                  {docsByCategory[category].length === 0 && (
                    <div className="p-3 text-xs text-gray-400">No files</div>
                  )}
                </div>
              )}
            </div>
          ))}
          
          {documents.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p className="text-sm">No documents uploaded yet</p>
              <Button 
                className="mt-4" 
                onClick={() => setIsUploadModalOpen(true)}
              >
                Upload Documents
              </Button>
            </div>
          )}
        </div>
        
        {/* Search Bar at bottom of left panel */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <SimpleSearchInput 
            onSearch={handleDealSearch}
            placeholder="Search across documents..."
            isLoading={isSearching}
          />
        </div>
      </div>

      {/* Center - Document Viewer */}
      <div className="flex-1 flex flex-col">
        {selectedDocument ? (
          <DocViewer document={selectedDocument} />
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <FileText className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <p>Select a document to view</p>
            </div>
          </div>
        )}
      </div>

      {/* Right Sidebar - Memos & Search Results */}
      <div className="w-96 bg-white border-l border-gray-200">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-2 rounded-none border-b">
            <TabsTrigger value="memos" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Memos
            </TabsTrigger>
            <TabsTrigger value="search-results" className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Search Results
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="memos" className="flex-1 flex flex-col mt-0">
            <MemoEditor dealId={dealId} />
          </TabsContent>
          
          <TabsContent value="search-results" className="flex-1 flex flex-col mt-0">
            <div className="flex-1 overflow-y-auto">
              <ResultsPane 
                results={searchResults}
                answer={answer}
                isSearching={isSearching}
                onResultClick={(result) => {
                  // Navigate to document and highlight text
                  setSelectedDocumentId(result.document_id)
                }}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>

      <UploadModal 
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        dealId={dealId}
      />
    </div>
  )
} 