import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { useUpload } from '@/hooks/useDocuments'
import { useDealStore } from '@/store/dealStore'
import { Upload, X, File, AlertCircle, CheckCircle } from 'lucide-react'

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  dealId?: string
}

interface FileUploadState {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

export function UploadModal({ isOpen, onClose, dealId }: UploadModalProps) {
  const [files, setFiles] = useState<FileUploadState[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [dealName, setDealName] = useState('')
  const [dealDescription, setDealDescription] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  
  const { uploadDocuments } = useUpload()
  const { createDeal } = useDealStore()
  const router = useRouter()

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const droppedFiles = Array.from(e.dataTransfer.files)
    addFiles(droppedFiles)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      addFiles(selectedFiles)
    }
  }

  const addFiles = (newFiles: File[]) => {
    const validFiles = newFiles.filter(file => {
      const isValidType = [
        'application/pdf', 
        'application/msword', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
        'text/plain',
        'text/csv',
        'application/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      ].includes(file.type)
      const isValidSize = file.size <= 50 * 1024 * 1024 // 50MB
      return isValidType && isValidSize
    })

    const fileStates: FileUploadState[] = validFiles.map(file => ({
      file,
      status: 'pending'
    }))

    setFiles(prev => [...prev, ...fileStates])
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    let targetDealId = dealId
    
    // Create deal if no dealId provided
    if (!dealId && dealName.trim()) {
      try {
        const newDeal = await createDeal(dealName.trim(), dealDescription.trim() || undefined)
        targetDealId = newDeal.id
      } catch (error) {
        console.error('Failed to create deal:', error)
        return
      }
    }

    if (!targetDealId) return

    setIsUploading(true)

    // Capture the current files BEFORE mutating state so we don't lose them due to React's async state updates
    const filesToUpload = files.map(f => f.file)

    // Mark files as uploading (state update happens asynchronously)
    setFiles(prev => prev.map(f => ({ ...f, status: 'uploading' })))

    console.log('=== UPLOAD MODAL DEBUG ===')
    console.log('Deal ID for upload:', targetDealId)
    console.log('Files to upload count:', filesToUpload.length)
    console.log('Files to upload details:', filesToUpload.map(f => ({ name: f.name, size: f.size, type: f.type })))

    try {
      // Upload all files in a single request using the captured list
      await uploadDocuments(targetDealId, filesToUpload)
      
      // Set all files to success status
      setFiles(prev => prev.map(f => ({ ...f, status: 'success' })))

      // Navigate to the deal page on success
      if (targetDealId) {
        router.push(`/deals/${targetDealId}`)
        onClose()
      }
    } catch (error) {
      console.error('Upload failed:', error)
      
      // Extract the error message from the Error object
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      
      // Set all files to error status with the specific error message
      setFiles(prev => prev.map(f => ({ 
        ...f, 
        status: 'error', 
        error: errorMessage 
      })))
    } finally {
      setIsUploading(false)
    }
  }

  const handleClose = () => {
    setFiles([])
    setDealName('')
    setDealDescription('')
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-lg font-semibold">
            {dealId ? 'Upload Documents' : 'Create New Deal'}
          </h2>
          <Button variant="ghost" size="icon" onClick={handleClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="p-6 space-y-6">
          {/* Deal Info (only if creating new deal) */}
          {!dealId && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Deal Name *</label>
                <input
                  type="text"
                  value={dealName}
                  onChange={(e) => setDealName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter deal name..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Description</label>
                <textarea
                  value={dealDescription}
                  onChange={(e) => setDealDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter deal description..."
                />
              </div>
            </div>
          )}

          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-900 mb-2">
              Drop files here or click to browse
            </p>
            <p className="text-sm text-gray-600 mb-4">
              Support for PDF, Word, CSV, Excel, and text files (max 50MB each)
            </p>
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.xls"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <Button asChild>
              <label htmlFor="file-upload" className="cursor-pointer">
                Select Files
              </label>
            </Button>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-medium">Selected Files</h3>
              {files.map((fileState, index) => (
                <Card key={index}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <File className="h-5 w-5 text-gray-400" />
                        <div>
                          <p className="text-sm font-medium">{fileState.file.name}</p>
                          <p className="text-xs text-gray-500">
                            {Math.round(fileState.file.size / 1024)} KB
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {fileState.status === 'success' && (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                        {fileState.status === 'error' && (
                          <AlertCircle className="h-5 w-5 text-red-500" />
                        )}
                        {fileState.status === 'uploading' && (
                          <div className="h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        )}
                        {fileState.status === 'pending' && (
                          <Button 
                            variant="ghost" 
                            size="icon"
                            onClick={() => removeFile(index)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                    {fileState.error && (
                      <p className="text-xs text-red-500 mt-2">{fileState.error}</p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 p-6 border-t">
          <Button variant="outline" onClick={handleClose}>
            Close
          </Button>
          <Button 
            onClick={uploadFiles}
            disabled={files.length === 0 || (!dealId && !dealName.trim()) || isUploading}
          >
            {isUploading ? 'Uploading...' : (dealId ? 'Upload Files' : 'Create Deal & Upload')}
          </Button>
        </div>
      </div>
    </div>
  )
} 