import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAoNL7Mu-2aDtSLen8puL_RYsSfxHZAZMo",
  authDomain: "eshani-97fb8.firebaseapp.com",
  projectId: "eshani-97fb8",
  storageBucket: "eshani-97fb8.firebasestorage.app",
  messagingSenderId: "182165103583",
  appId: "1:182165103583:web:19fc376cb447220ddbda6a",
};

const app = initializeApp(firebaseConfig);

export const db = getFirestore(app);