import { initializeApp, getApps, FirebaseApp } from 'firebase/app';
import { getAuth, Auth } from 'firebase/auth';

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

// Initialize Firebase - lazy initialization to avoid SSR issues
let app: FirebaseApp | null = null;
let firebaseAuth: Auth | null = null;

function initializeFirebase() {
  if (typeof window === 'undefined') return null;
  
  if (!app) {
    app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
  }
  
  if (!firebaseAuth && app) {
    firebaseAuth = getAuth(app);
  }
  
  return { app, auth: firebaseAuth };
}

// Export getter functions instead of direct instances
export const getFirebaseAuth = () => {
  const firebase = initializeFirebase();
  return firebase?.auth || null;
};

export const getFirebaseApp = () => {
  const firebase = initializeFirebase();
  return firebase?.app || null;
};

// For backward compatibility, export auth that initializes lazily
export const auth = new Proxy({} as Auth, {
  get(target, prop) {
    const firebaseAuth = getFirebaseAuth();
    if (!firebaseAuth) {
      throw new Error('Firebase not initialized - make sure you are on the client side');
    }
    return firebaseAuth[prop as keyof Auth];
  }
});

export default getFirebaseApp();