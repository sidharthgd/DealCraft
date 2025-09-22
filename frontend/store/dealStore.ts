import { create } from 'zustand'
import { Deal, Document, Memo } from '@/types'
import { api } from '@/lib/api'

interface DealStore {
  deals: Deal[]
  documents: Record<string, Document[]>
  memos: Record<string, Memo[]>
  
  // Actions
  fetchDeals: () => Promise<void>
  fetchDeal: (dealId: string) => Promise<Deal | null>
  createDeal: (name: string, description?: string) => Promise<Deal>
  deleteDeal: (dealId: string) => Promise<void>
  fetchDocuments: (dealId: string) => Promise<void>
  addDocument: (dealId: string, document: Document) => void
  removeDocument: (documentId: string) => void
  updateDealDocumentCount: (dealId: string, count: number) => void
  fetchMemos: (dealId: string) => Promise<void>
  addMemo: (dealId: string, memo: Memo) => void
  updateMemo: (memo: Memo) => void
  removeMemo: (memoId: string) => void
}

export const useDealStore = create<DealStore>((set, get) => ({
  deals: [],
  documents: {},
  memos: {},

  fetchDeals: async () => {
    try {
      const response = await api.getDeals()
      const deals = response.data
      set({ deals })
    } catch (error) {
      console.error('Failed to fetch deals:', error)
    }
  },

  fetchDeal: async (dealId: string) => {
    try {
      // First check if deal is already in store
      const existingDeal = get().deals.find(d => d.id === dealId)
      if (existingDeal) {
        return existingDeal
      }

      // If not found, fetch from API
      const response = await api.getDeal(dealId)
      const deal = response.data
      
      // Add to store
      set((state) => ({ 
        deals: [...state.deals.filter(d => d.id !== dealId), deal] 
      }))
      
      return deal
    } catch (error) {
      console.error('Failed to fetch deal:', error)
      return null
    }
  },

  createDeal: async (name: string, description?: string) => {
    try {
      const response = await api.createDeal(name, description || '')
      const deal = response.data
      set((state) => ({ deals: [...state.deals, deal] }))
      return deal
    } catch (error) {
      console.error('Failed to create deal:', error)
      throw error
    }
  },

  deleteDeal: async (dealId: string) => {
    try {
      await api.deleteDeal(dealId)
      set((state) => ({
        deals: state.deals.filter(d => d.id !== dealId)
      }))
    } catch (error) {
      console.error('Failed to delete deal:', error)
      throw error
    }
  },

  fetchDocuments: async (dealId: string) => {
    try {
      const response = await api.getDocuments(dealId)
      const documents = response.data
      set((state) => ({
        documents: { ...state.documents, [dealId]: documents }
      }))
    } catch (error) {
      console.error('Failed to fetch documents:', error)
    }
  },

  addDocument: (dealId: string, document: Document) => {
    set((state) => {
      const updatedDocuments = {
        ...state.documents,
        [dealId]: [...(state.documents[dealId] || []), document]
      }
      
      // Update the deal's document count
      const updatedDeals = state.deals.map(deal => 
        deal.id === dealId 
          ? { ...deal, document_count: updatedDocuments[dealId].length }
          : deal
      )
      
      return { 
        documents: updatedDocuments,
        deals: updatedDeals
      }
    })
  },

  removeDocument: (documentId: string) => {
    set((state) => {
      const newDocuments = { ...state.documents }
      let affectedDealId: string | null = null
      
      Object.keys(newDocuments).forEach((dealId) => {
        const filteredDocs = newDocuments[dealId].filter(
          (doc) => doc.id !== documentId
        )
        if (filteredDocs.length !== newDocuments[dealId].length) {
          affectedDealId = dealId
        }
        newDocuments[dealId] = filteredDocs
      })
      
      // Update the deal's document count if a document was removed
      const updatedDeals = affectedDealId 
        ? state.deals.map(deal => 
            deal.id === affectedDealId 
              ? { ...deal, document_count: newDocuments[affectedDealId].length }
              : deal
          )
        : state.deals
      
      return { 
        documents: newDocuments,
        deals: updatedDeals
      }
    })
  },

  updateDealDocumentCount: (dealId: string, count: number) => {
    set((state) => ({
      deals: state.deals.map(deal => 
        deal.id === dealId 
          ? { ...deal, document_count: count }
          : deal
      )
    }))
  },

  fetchMemos: async (dealId: string) => {
    try {
      const response = await api.getMemos(dealId)
      const memos = response.data
      set((state) => ({
        memos: { ...state.memos, [dealId]: memos }
      }))
    } catch (error) {
      console.error('Failed to fetch memos:', error)
    }
  },

  addMemo: (dealId: string, memo: Memo) => {
    set((state) => ({
      memos: {
        ...state.memos,
        [dealId]: [...(state.memos[dealId] || []), memo]
      }
    }))
  },

  updateMemo: (memo: Memo) => {
    set((state) => {
      const newMemos = { ...state.memos }
      Object.keys(newMemos).forEach((dealId) => {
        const memoIndex = newMemos[dealId].findIndex((m) => m.id === memo.id)
        if (memoIndex !== -1) {
          newMemos[dealId][memoIndex] = memo
        }
      })
      return { memos: newMemos }
    })
  },

  removeMemo: (memoId: string) => {
    set((state) => {
      const newMemos = { ...state.memos }
      Object.keys(newMemos).forEach((dealId) => {
        newMemos[dealId] = newMemos[dealId].filter((memo) => memo.id !== memoId)
      })
      return { memos: newMemos }
    })
  },
})) 