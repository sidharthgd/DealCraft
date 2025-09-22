import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import '../styles/react-pdf.css'
import { QueryProvider } from '../components/providers/QueryProvider'
import { AuthProvider } from '../components/providers/AuthProvider'
import { AuthGuard } from '../components/auth/AuthGuard'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'DealCraft AI',
  description: 'AI-powered deal document analysis platform',
  icons: {
    icon: '/favicon.svg',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Suppress Google OAuth COOP console errors (these are Google/browser-related and don't affect functionality)
              const originalConsoleError = console.error;
              console.error = function(...args) {
                const message = args[0]?.toString() || '';
                if (
                  message.includes('Cross-Origin-Opener-Policy') || 
                  message.includes('window.closed call') ||
                  message.includes('window.postMessage call')
                ) {
                  return; // Suppress these specific COOP errors
                }
                originalConsoleError.apply(console, args);
              };
            `,
          }}
        />
      </head>
      <body className={inter.className}>
        <AuthProvider>
          <QueryProvider>
            <AuthGuard>
              {children}
            </AuthGuard>
          </QueryProvider>
        </AuthProvider>
      </body>
    </html>
  )
} 