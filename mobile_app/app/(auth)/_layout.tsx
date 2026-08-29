import { Redirect, Stack } from 'expo-router';

import { useAuth } from '../auth';

export default function AuthLayout() {
  const auth = useAuth();

  // Already signed in? Skip the auth screens.
  if (auth.ready && auth.isAuthed) {
    return <Redirect href="/(tabs)" />;
  }

  return <Stack screenOptions={{ headerShown: false }} />;
}
