import { useState, useEffect, useCallback, useRef } from 'react'
import { Search, Loader2 } from 'lucide-react'

interface SimpleSearchInputProps {
  onSearch: (query: string) => void
  placeholder?: string
  isLoading?: boolean
}

export function SimpleSearchInput({ onSearch, placeholder = "Search documents...", isLoading = false }: SimpleSearchInputProps) {
  const [query, setQuery] = useState('')
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Debounced search function
  const debouncedSearch = useCallback(
    (searchQuery: string) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      timeoutRef.current = setTimeout(() => {
        if (searchQuery.trim()) {
          onSearch(searchQuery.trim())
        }
      }, 500)
    },
    [onSearch]
  )

  // Trigger search when query changes
  useEffect(() => {
    if (query.trim()) {
      debouncedSearch(query)
    }
    
    // Cleanup function to cancel pending timeout
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [query, debouncedSearch])

  return (
    <div className="relative">
      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
      {isLoading && (
        <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-gray-400" />
      )}
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-10 pr-10 py-3 bg-gray-100 border-0 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white text-sm"
        disabled={isLoading}
      />
    </div>
  )
} 