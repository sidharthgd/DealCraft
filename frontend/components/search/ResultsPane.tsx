import { SearchResult } from '@/types'
import { Card, CardContent } from '@/components/ui/card'
import { FileText, ArrowRight } from 'lucide-react'

interface ResultsPaneProps {
  results: SearchResult[]
  answer?: string
  isSearching?: boolean
  onResultClick?: (result: SearchResult) => void
}

export function ResultsPane({ results, answer, isSearching = false, onResultClick }: ResultsPaneProps) {
  if (!isSearching && results.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p className="text-sm">No results found</p>
      </div>
    )
  }

  return (
    <div className="space-y-3 p-4">
      {/* AI Answer */}
      {answer !== undefined && (
        <div className="border border-gray-200 rounded-md p-4 bg-gray-50 text-sm text-gray-800">
          {isSearching ? 'Loading answer…' : answer || 'No answer generated'}
        </div>
      )}

      {results.map((result, index) => (
        <Card 
          key={`${result.document_id}-${result.chunk_index}-${index}`}
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => onResultClick?.(result)}
        >
          <CardContent className="p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-900 truncate">
                  {result.document_name}
                </span>
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <span>{Math.round(result.similarity_score * 100)}%</span>
                <ArrowRight className="h-3 w-3" />
              </div>
            </div>
            
            <p className="text-sm text-gray-700 line-clamp-3 leading-relaxed">
              {result.content}
            </p>
            
            <div className="mt-2 text-xs text-gray-500">
              Chunk {result.chunk_index + 1} • Characters {result.start_char}-{result.end_char}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
} 