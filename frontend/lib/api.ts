import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { auth } from './firebase';

// Create axios instance with base configuration
const createApiClient = (): AxiosInstance => {
  // In production, use the direct backend URL. In development, use relative URLs for Next.js proxy
  const isDevelopment = process.env.NODE_ENV === 'development'
  const isBrowser = typeof window !== 'undefined'
  
  let baseURL: string
  
  if (isDevelopment && isBrowser) {
    // Development: use relative URL for Next.js proxy
    baseURL = '/api/v1'
  } else {
    // Production or SSR: use direct backend URL
    baseURL = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1`
  }

  const client = axios.create({
    baseURL: baseURL,
    timeout: 1800000, // 30 minutes default to prevent client aborts on slow endpoints
    maxContentLength: -1,
    maxBodyLength: -1,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Add request interceptor to include authentication token
  client.interceptors.request.use(
    async (config) => {
      try {
        const user = auth.currentUser;
        if (user) {
          const token = await user.getIdToken();
          config.headers.Authorization = `Bearer ${token}`;
        }
      } catch (error) {
        console.warn('Failed to get authentication token:', error);
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Add response interceptor to handle common errors
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Handle unauthorized access
        console.error('Unauthorized access - user may need to log in again');
        // Optionally redirect to login or refresh token
      }
      return Promise.reject(error);
    }
  );

  return client;
};

// Global API client instance
export const apiClient = createApiClient();

// Utility functions for common API operations
export const api = {
  // Upload files
  uploadFiles: async (files: FileList, dealName: string) => {
    const formData = new FormData();
    
    Array.from(files).forEach((file) => {
      formData.append('files', file);
    });
    formData.append('deal_name', dealName);

    return apiClient.post('/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Large files + server-side processing; allow up to 30 minutes
      timeout: 1800000,
      maxContentLength: -1,
      maxBodyLength: -1,
    });
  },

  // Search documents
  searchDocuments: async (query: string, dealId?: string) => {
    const params: any = { query };
    if (dealId) params.deal_id = dealId;
    
    return apiClient.get('/search/', { params });
  },

  // Generate memo
  generateMemo: async (dealId: string, customFields?: { discussion_date?: string; ftf_equity_size?: string; expected_closing?: string }) => {
    const payload = { deal_id: dealId, ...customFields };
    return apiClient.post('/memo/generate', payload, { timeout: 600000 });
  },

  // Get deals
  getDeals: async () => {
    return apiClient.get('/deals/');
  },

  // Get deal by ID
  getDeal: async (dealId: string) => {
    return apiClient.get(`/deals/${dealId}`);
  },

  // Get documents for a deal
  getDocuments: async (dealId: string) => {
    return apiClient.get(`/deals/${dealId}/documents`);
  },

  // Get memos for a deal
  getMemos: async (dealId: string) => {
    return apiClient.get(`/deals/${dealId}/memos`);
  },

  // Create memo
  createMemo: async (dealId: string, title: string, content: string) => {
    return apiClient.post('/memo/', { deal_id: dealId, title, content });
  },

  // Update memo
  updateMemo: async (memoId: string, title: string, content: string) => {
    return apiClient.put(`/memo/${memoId}`, { title, content });
  },

  // Delete memo
  deleteMemo: async (memoId: string) => {
    return apiClient.delete(`/memo/${memoId}`);
  },

  // Download memo file
  downloadMemoFile: async (filename: string) => {
    return apiClient.get(`/memo/download/${filename}`, { responseType: 'blob' });
  },

  // Upload single document
  uploadDocument: async (dealId: string, file: File) => {
    const formData = new FormData();
    formData.append('files', file);
    formData.append('deal_id', dealId);
    
    return apiClient.post('/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 1800000, // 30 minutes
      maxContentLength: -1,
      maxBodyLength: -1,
    });
  },

  // Upload multiple documents
  uploadDocuments: async (dealId: string, files: File[]) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('deal_id', dealId);
    
    return apiClient.post('/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 1800000, // 30 minutes
      maxContentLength: -1,
      maxBodyLength: -1,
    });
  },

  // Create deal
  createDeal: async (name: string, description: string) => {
    return apiClient.post('/deals/', { name, description });
  },

  // Delete deal
  deleteDeal: async (dealId: string) => {
    return apiClient.delete(`/deals/${dealId}`);
  },

  // Search
  search: async (dealId: string, query: string) => {
    return apiClient.get('/search/', { params: { deal_id: dealId, query } });
  },
};

export default api;