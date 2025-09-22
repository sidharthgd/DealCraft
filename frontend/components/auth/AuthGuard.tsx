'use client';

import React, { useState } from 'react';
import { useAuth } from '../providers/AuthProvider';
import { LoginForm } from './LoginForm';

interface AuthGuardProps {
  children: React.ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { user, loading } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);

  // Show loading spinner while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  // Show login form if user is not authenticated
  if (!user) {
    return (
      <LoginForm 
        onToggleMode={() => setIsSignUp(!isSignUp)} 
        isSignUp={isSignUp} 
      />
    );
  }

  // User is authenticated, show the protected content
  return <>{children}</>;
}