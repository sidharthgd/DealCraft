import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { SearchResult } from '@/types'
import { api } from '@/lib/api'

export function useSearch() {
  const [query, setQuery] = useState('')
  const [dealId, setDealId] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [answer, setAnswer] = useState<string>('')
  const [currentSearchDealId, setCurrentSearchDealId] = useState('')

  const { isLoading: isSearching, data, refetch } = useQuery({
    queryKey: ['search', currentSearchDealId, query],
    queryFn: async () => {
      if (!currentSearchDealId || !query) return null
      const response = await api.search(currentSearchDealId, query)
      return response.data
    },
    enabled: false,
  })

  useEffect(() => {
    if (data) {
      setSearchResults(data.results)
      setAnswer(data.answer || '')
    }
  }, [data])

  const search = async (searchQuery: string, searchDealId?: string) => {
    const targetDealId = searchDealId || dealId
    if (searchDealId) setDealId(searchDealId)
    setQuery(searchQuery)
    setCurrentSearchDealId(targetDealId)
    if (targetDealId && searchQuery) {
      refetch()
    }
  }

  const clearSearch = () => {
    setQuery('')
    setSearchResults([])
    setAnswer('')
  }

  return {
    query,
    searchResults,
    answer,
    isSearching,
    search,
    clearSearch,
  }
} 