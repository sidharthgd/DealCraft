import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useDealStore } from '@/store/dealStore'
import { api } from '@/lib/api'

export function useDocuments(dealId: string) {
  const { documents, fetchDocuments, addDocument, removeDocument } = useDealStore()
  
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['documents', dealId],
    queryFn: async () => {
      const response = await api.getDocuments(dealId)
      return response.data
    },
    enabled: !!dealId,
  })

  useEffect(() => {
    if (data) {
      fetchDocuments(dealId)
    }
  }, [data, dealId, fetchDocuments])

  return {
    documents: documents[dealId] || [],
    isLoading,
    error,
    refetch,
    addDocument: (document: any) => addDocument(dealId, document),
    removeDocument,
  }
}

export function useUpload() {
  const { addDocument } = useDealStore()

  const uploadDocument = async (dealId: string, file: File) => {
    try {
      const response = await api.uploadDocument(dealId, file)
      const document = response.data
      addDocument(dealId, document)
      return document
    } catch (error) {
      throw error
    }
  }

  const uploadDocuments = async (dealId: string, files: File[]) => {
    try {
      const response = await api.uploadDocuments(dealId, files)
      const documents = response.data
      documents.forEach((document: any) => addDocument(dealId, document))
      return documents
    } catch (error) {
      throw error
    }
  }

  return { uploadDocument, uploadDocuments }
} 