import { useState, useEffect } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Placeholder from '@tiptap/extension-placeholder'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useDealStore } from '@/store/dealStore'
import { api } from '@/lib/api'
import { Memo } from '@/types'
import { Plus, Edit3, Save, Trash2, FileText, Sparkles, Download, Upload } from 'lucide-react'

interface MemoEditorProps {
  dealId: string
}

export function MemoEditor({ dealId }: MemoEditorProps) {
  const [memos, setMemos] = useState<Memo[]>([])
  const [selectedMemo, setSelectedMemo] = useState<Memo | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [title, setTitle] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  
  // Custom fields for memo generation
  const [discussionDate, setDiscussionDate] = useState('')
  const [ftfEquitySize, setFtfEquitySize] = useState('')
  const [expectedClosing, setExpectedClosing] = useState('')
  const [orgChartImageBase64, setOrgChartImageBase64] = useState<string>('')


  // Get documents from store for memo generation
  const { documents } = useDealStore()
  const dealDocuments = documents[dealId] || []

  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: 'Start writing your memo...',
      }),
    ],
    content: '',
    immediatelyRender: false,
    onUpdate: ({ editor }) => {
      // Auto-save could be implemented here
    },
  })

  // Load memos for the deal
  useEffect(() => {
    loadMemos()
  }, [dealId])

  const loadMemos = async () => {
    try {
      const response = await api.getMemos(dealId)
      setMemos(response.data)
    } catch (error) {
      console.error('Failed to load memos:', error)
    }
  }

  const startNewMemo = () => {
    setSelectedMemo(null)
    setTitle('')
    setIsEditing(true)
    editor?.commands.setContent('')
  }

  const editMemo = (memo: Memo) => {
    setSelectedMemo(memo)
    setTitle(memo.title)
    setIsEditing(true)
    editor?.commands.setContent(memo.content)
  }

  const saveMemo = async () => {
    if (!title.trim() || !editor) return

    setIsLoading(true)
    try {
      const content = editor.getHTML()
      
      if (selectedMemo) {
        // Update existing memo
        const response = await api.updateMemo(selectedMemo.id, title.trim(), content)
        const updatedMemo = response.data
        setMemos(prev => prev.map(m => m.id === updatedMemo.id ? updatedMemo : m))
        setSelectedMemo(updatedMemo)
      } else {
        // Create new memo
        const response = await api.createMemo(dealId, title.trim(), content)
        const newMemo = response.data
        setMemos(prev => [...prev, newMemo])
        setSelectedMemo(newMemo)
      }
      
      setIsEditing(false)
    } catch (error) {
      console.error('Failed to save memo:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const [generatedFile, setGeneratedFile] = useState<{
    title: string
    filename: string
    generated_at: string
    document_count: number
    file_size: number
    message: string
  } | null>(null)

  const generateMemo = async () => {
    if (dealDocuments.length === 0) {
      alert('No documents available for memo generation. Please upload documents first.')
      return
    }

    setIsGenerating(true)
    try {
      const documentIds = dealDocuments.map(doc => doc.id)
      
      // Helper: format yyyy-mm-dd to mm/dd/yyyy
      const toMMDDYYYY = (value: string): string | undefined => {
        const v = (value || '').trim()
        if (!v) return undefined
        // Expect browser date input format yyyy-mm-dd
        const parts = v.split('-')
        if (parts.length === 3) {
          const [yyyy, mm, dd] = parts
          const mm2 = mm.padStart(2, '0')
          const dd2 = dd.padStart(2, '0')
          return `${mm2}/${dd2}/${yyyy}`
        }
        // Fallback: try Date parsing
        const d = new Date(v)
        if (!isNaN(d.getTime())) {
          const mm2 = String(d.getMonth() + 1).padStart(2, '0')
          const dd2 = String(d.getDate()).padStart(2, '0')
          const yyyy = String(d.getFullYear())
          return `${mm2}/${dd2}/${yyyy}`
        }
        return v
      }

      // Prepare custom fields
      const customFields: Record<string, string | undefined> = {
        discussion_date: toMMDDYYYY(discussionDate),
        ftf_equity_size: ftfEquitySize.trim() || undefined,
        expected_closing: toMMDDYYYY(expectedClosing),
        // This will be merged at the top level by the API helper (payload body) if implemented as a spread.
        // Backend expects this field at the top level; most existing helper merges fields.
        org_chart_image_base64: orgChartImageBase64 || undefined,
      }
      
      const response = await api.generateMemo(dealId, customFields)
      const result = response.data
      
      // Set the generated file info instead of content
      setGeneratedFile(result)
      setSelectedMemo(null)
      setIsEditing(false)
    } catch (error) {
      console.error('Failed to generate memo:', error)
      alert('Failed to generate memo. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  const downloadGeneratedMemo = async () => {
    if (!generatedFile) return

    try {
      const response = await api.downloadMemoFile(generatedFile.filename)
      const blob = response.data
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = generatedFile.filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Failed to download memo:', error)
      alert('Failed to download memo. Please try again.')
    }
  }

  const downloadMemoContent = (memo: Memo) => {
    // Convert HTML content to plain text and download as PDF
    const tempDiv = document.createElement('div')
    tempDiv.innerHTML = memo.content
    const plainTextContent = tempDiv.textContent || tempDiv.innerText || ''
    
    const content = `${memo.title}\n${'='.repeat(memo.title.length)}\n\n${plainTextContent}`
    
    const blob = new Blob([content], { type: 'text/plain' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${memo.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  const deleteMemo = async (memoId: string) => {
    if (!confirm('Are you sure you want to delete this memo?')) return

    try {
      await api.deleteMemo(memoId)
      setMemos(prev => prev.filter(m => m.id !== memoId))
      if (selectedMemo?.id === memoId) {
        setSelectedMemo(null)
        setIsEditing(false)
      }
    } catch (error) {
      console.error('Failed to delete memo:', error)
    }
  }

  const cancelEdit = () => {
    setIsEditing(false)
    if (selectedMemo) {
      setTitle(selectedMemo.title)
      editor?.commands.setContent(selectedMemo.content)
    } else {
      setTitle('')
      editor?.commands.setContent('')
      setSelectedMemo(null)
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium">Memos</h3>
        </div>

        {/* Memo List */}
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {memos.map((memo) => (
            <div
              key={memo.id}
              className={`p-2 rounded border cursor-pointer hover:bg-gray-50 ${
                selectedMemo?.id === memo.id ? 'bg-blue-50 border-blue-200' : 'border-gray-200'
              }`}
              onClick={() => editMemo(memo)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <FileText className="h-4 w-4 text-gray-400" />
                  <span className="text-sm font-medium truncate">{memo.title}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation()
                      downloadMemoContent(memo)
                    }}
                    title="Download memo as .pdf"
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteMemo(memo.id)
                    }}
                    title="Delete memo"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {new Date(memo.updated_at).toLocaleDateString()}
              </p>
            </div>
          ))}
          
          {memos.length === 0 && (
            <div className="text-center py-4 text-gray-500">
              <FileText className="h-8 w-8 mx-auto mb-2 text-gray-300" />
              <p className="text-sm">No memos yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 flex flex-col">
        {(isEditing || selectedMemo) ? (
          <>
            {/* Title Input */}
            {isEditing && (
              <div className="p-4 border-b border-gray-200">
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Memo title..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}

            {/* Editor Content - Show when viewing/editing memo or editing new memo */}
            {(selectedMemo && !isEditing) || (selectedMemo && isEditing) || (isEditing && !selectedMemo) ? (
              <div className="flex-1 overflow-hidden">
                <div className="h-full p-4">
                  <div className="h-full border border-gray-200 rounded-md overflow-hidden">
                    <EditorContent 
                      editor={editor} 
                      className="h-full prose prose-sm max-w-none p-4 focus:outline-none"
                    />
                  </div>
                </div>
              </div>
            ) : null}

            {/* Editor Actions - New memo creation */}
            {isEditing && !selectedMemo && (
              <div className="p-4 border-t border-gray-200">
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={cancelEdit}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={saveMemo}
                    disabled={!title.trim() || isLoading}
                    className="flex items-center gap-2"
                  >
                    <Save className="h-4 w-4" />
                    {isLoading ? 'Saving...' : 'Save memo'}
                  </Button>
                </div>
              </div>
            )}

            {/* Editor Actions - Existing memo editing */}
            {isEditing && selectedMemo && (
              <div className="p-4 border-t border-gray-200">
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={cancelEdit}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={saveMemo}
                    disabled={!title.trim() || isLoading}
                  >
                    <Save className="h-4 w-4 mr-2" />
                    {isLoading ? 'Saving...' : 'Save'}
                  </Button>
                </div>
              </div>
            )}

            {/* View Mode Actions */}
            {!isEditing && selectedMemo && (
              <div className="p-4 border-t border-gray-200">
                <div className="flex justify-end">
                  <Button size="sm" onClick={() => setIsEditing(true)}>
                    <Edit3 className="h-4 w-4 mr-2" />
                    Edit
                  </Button>
                </div>
              </div>
            )}
          </>
        ) : generatedFile ? (
          <div className="flex-1 flex items-center justify-center p-4">
            <Card className="w-full max-w-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-green-500" />
                  Memo Generated Successfully!
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-sm text-gray-600">
                  <p><strong>File:</strong> {generatedFile.filename}</p>
                  <p><strong>Generated:</strong> {new Date(generatedFile.generated_at).toLocaleString()}</p>
                  <p><strong>Documents processed:</strong> {generatedFile.document_count}</p>
                  <p><strong>File size:</strong> {(generatedFile.file_size / 1024).toFixed(1)} KB</p>
                </div>
                
                <div className="flex gap-2">
                  <Button 
                    onClick={downloadGeneratedMemo}
                    size="sm"
                    className="flex items-center gap-1 flex-1"
                  >
                    <Download className="h-3 w-3" />
                    Download
                  </Button>
                  <Button 
                    variant="outline"
                    size="sm"
                    onClick={() => setGeneratedFile(null)}
                    className="flex items-center gap-1 whitespace-nowrap"
                  >
                    <Plus className="h-3 w-3" />
                    Generate New
                  </Button>
                </div>
                
                <p className="text-xs text-gray-500 text-center">
                  {generatedFile.message}
                </p>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <div className="text-center w-full max-w-md px-4">
              <FileText className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <p className="mb-6">No memo selected</p>
              
              {/* Custom fields for memo generation */}
              <div className="space-y-4 mb-6">
                <div className="text-left">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Discussion Date:
                  </label>
                  <Input
                    type="date"
                    value={discussionDate}
                    onChange={(e) => setDiscussionDate(e.target.value)}
                    className="w-full"
                  />
                </div>
                
                <div className="text-left">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    FTF Equity Size:
                  </label>
                  <Input
                    type="text"
                    placeholder="Enter FTF equity size"
                    value={ftfEquitySize}
                    onChange={(e) => setFtfEquitySize(e.target.value)}
                    className="w-full"
                  />
                </div>
                
                <div className="text-left">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Expected Closing:
                  </label>
                  <Input
                    type="date"
                    value={expectedClosing}
                    onChange={(e) => setExpectedClosing(e.target.value)}
                    className="w-full"
                  />
                </div>

                <div className="text-left">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Organization Chart Image (PNG/JPEG/SVG):
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg,image/svg+xml"
                      onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (!file) {
                          setOrgChartImageBase64('')
                          return
                        }
                        const reader = new FileReader()
                        reader.onload = () => {
                          const result = reader.result as string
                          setOrgChartImageBase64(result)
                        }
                        reader.readAsDataURL(file)
                      }}
                      className="hidden"
                      id="org-chart-upload"
                    />
                    <label 
                      htmlFor="org-chart-upload" 
                      className="flex items-center justify-center gap-2 w-full px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-colors"
                    >
                      <Upload className="h-5 w-5 text-gray-400" />
                      <span className="text-sm text-gray-600">
                        {orgChartImageBase64 ? 'Change image' : 'Click to upload image'}
                      </span>
                    </label>
                  </div>
                  {orgChartImageBase64 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 mb-1">Preview:</p>
                      <img
                        src={orgChartImageBase64}
                        alt="Org Chart Preview"
                        className="max-h-40 rounded border"
                      />
                    </div>
                  )}
                </div>
              </div>
              
              <Button 
                onClick={generateMemo}
                disabled={isGenerating || dealDocuments.length === 0}
                className="flex items-center gap-2"
              >
                <Sparkles className="h-4 w-4" />
                {isGenerating ? 'Generating...' : 'Generate with AI'}
              </Button>
              {dealDocuments.length === 0 && (
                <p className="text-xs text-gray-400 mt-2">
                  Upload documents to enable AI generation
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 