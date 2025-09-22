'use client'

import { useState, useEffect } from 'react'
import { Document as PDFDocument, Page, pdfjs } from 'react-pdf'
import { Document } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { FileText, Download, Eye, ZoomIn, ZoomOut, ChevronLeft, ChevronRight } from 'lucide-react'

// Set up PDF.js worker - use local worker instead of CDN to avoid CORS issues
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js'

interface DocViewerProps {
  document: Document
}

export function DocViewer({ document }: DocViewerProps) {
  const [content, setContent] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [zoom, setZoom] = useState(100)
  const [numPages, setNumPages] = useState<number | null>(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [pdfError, setPdfError] = useState<string | null>(null)

  useEffect(() => {
    if (document.content_type !== 'application/pdf') {
      // For non-PDF documents, simulate loading content
      setIsLoading(true)
      setTimeout(() => {
        setContent(`This is the content of ${document.name}.\n\nThis is a placeholder view. In a real implementation, you would:\n\n1. Fetch the actual document content from the backend\n2. Display text content with proper formatting\n3. Handle different file types appropriately\n\nDocument details:\n- Name: ${document.name}\n- Size: ${document.file_size} bytes\n- Type: ${document.content_type}\n- Created: ${document.created_at}`)
        setIsLoading(false)
      }, 1000)
    } else {
      setIsLoading(false)
    }
  }, [document])

  const handleDownload = () => {
    // In a real app, you'd download from the API
    console.log('Download document:', document.id)
  }

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 25, 200))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 25, 50))
  }

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setPageNumber(1)
    setPdfError(null)
  }

  const onDocumentLoadError = (error: Error) => {
    console.error('Error loading PDF:', error)
    setPdfError('Failed to load PDF document')
  }

  const goToPrevPage = () => {
    setPageNumber(prev => Math.max(prev - 1, 1))
  }

  const goToNextPage = () => {
    setPageNumber(prev => Math.min(prev + 1, numPages || 1))
  }

  // Generate a sample PDF URL - in production, this would come from your API
  const getPdfUrl = () => {
    // For demo purposes, use a sample PDF URL
    // In production, this would be: `/api/documents/${document.id}/download` or similar
    
    // You can use any publicly accessible PDF for testing, or use one from your test-files
    // For demonstration, using a sample PDF from PDF.js examples
    const samplePdfUrl = 'https://mozilla.github.io/pdf.js/web/compressed.tracemonkey-pldi-09.pdf'
    
    // In production, replace this with your actual API endpoint:
    // return `/api/documents/${document.id}/content`
    
    return samplePdfUrl
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading document...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-gray-400" />
            <div>
              <h3 className="font-medium text-gray-900">{document.name}</h3>
              <p className="text-sm text-gray-500">
                {Math.round(document.file_size / 1024)} KB • {document.content_type}
                {numPages && ` • ${numPages} pages`}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {document.content_type === 'application/pdf' && numPages && (
              <>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={goToPrevPage}
                  disabled={pageNumber <= 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-gray-600 min-w-[4rem] text-center">
                  {pageNumber} / {numPages}
                </span>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={goToNextPage}
                  disabled={pageNumber >= numPages}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <div className="w-px h-6 bg-gray-300 mx-2" />
              </>
            )}
            <Button variant="outline" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <span className="text-sm text-gray-600 min-w-[3rem] text-center">
              {zoom}%
            </span>
            <Button variant="outline" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto p-4">
        <Card className="h-full">
          <CardContent className="p-6">
            {document.content_type === 'application/pdf' ? (
              <div className="flex flex-col items-center">
                {pdfError ? (
                  <div className="text-center py-8">
                    <Eye className="h-16 w-16 text-red-300 mx-auto mb-4" />
                    <p className="text-red-600 mb-4">Error loading PDF</p>
                    <p className="text-sm text-gray-500">{pdfError}</p>
                  </div>
                ) : (
                  <PDFDocument
                    file={getPdfUrl()}
                    onLoadSuccess={onDocumentLoadSuccess}
                    onLoadError={onDocumentLoadError}
                    loading={
                      <div className="text-center py-8">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                        <p className="text-gray-600">Loading PDF...</p>
                      </div>
                    }
                  >
                    <Page
                      pageNumber={pageNumber}
                      scale={zoom / 100}
                      loading={
                        <div className="text-center py-4">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto"></div>
                        </div>
                      }
                      className="shadow-lg"
                    />
                  </PDFDocument>
                )}
              </div>
            ) : (
              <div 
                className="prose max-w-none"
                style={{ fontSize: `${zoom}%` }}
              >
                <pre className="whitespace-pre-wrap font-sans text-gray-800 leading-relaxed">
                  {content}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 