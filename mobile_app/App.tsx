import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import type { RootStackParamList } from './src/navigation/types';
import { AiAvatarScreen } from './src/screens/AiAvatarScreen';
import { AiTryOnScreen } from './src/screens/AiTryOnScreen';
import { AvatarCreatorScreen } from './src/screens/AvatarCreatorScreen';
import { ClassicSetupScreen } from './src/screens/ClassicSetupScreen';
import { DressPhotoScreen } from './src/screens/DressPhotoScreen';
import { EmailScreen } from './src/screens/EmailScreen';
import { FemaleAvatarScreen } from './src/screens/FemaleAvatarScreen';
import { FacePreviewScreen } from './src/screens/FacePreviewScreen';
import { FinalizedAvatarScreen } from './src/screens/FinalizedAvatarScreen';
import { GenderSelectScreen } from './src/screens/GenderSelectScreen';
import { MaleAvatarScreen } from './src/screens/MaleAvatarScreen';
import { ProfileScreen } from './src/screens/ProfileScreen';
import { WardrobeScreen } from './src/screens/WardrobeScreen';
import { colors } from './src/theme';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerStyle: { backgroundColor: colors.background },
            headerTintColor: colors.text,
            headerTitleStyle: { fontWeight: '700' },
            headerShadowVisible: false,
            contentStyle: { backgroundColor: colors.background },
          }}
        >
          <Stack.Screen name="Profile" component={ProfileScreen} options={{ title: 'eWardrobe' }} />
          <Stack.Screen name="DressPhoto" component={DressPhotoScreen} options={{ title: 'Add a clothing photo' }} />
          <Stack.Screen
            name="AiTryOn"
            component={AiTryOnScreen}
            options={{ title: 'Creating your avatar', headerBackVisible: false, gestureEnabled: false }}
          />
          <Stack.Screen name="AiAvatar" component={AiAvatarScreen} options={{ title: 'Your AI avatar' }} />
          <Stack.Screen
            name="ClassicSetup"
            component={ClassicSetupScreen}
            options={{ title: 'Setting up your avatar', headerBackVisible: false, gestureEnabled: false }}
          />
          <Stack.Screen name="Email" component={EmailScreen} options={{ title: 'Your email' }} />
          <Stack.Screen name="GenderSelect" component={GenderSelectScreen} options={{ title: 'Choose avatar' }} />
          <Stack.Screen
            name="AvatarCreator"
            component={AvatarCreatorScreen}
            options={{ title: 'Create your 3D avatar' }}
          />
          <Stack.Screen
            name="FacePreview"
            component={FacePreviewScreen}
            options={{ title: 'Face preview', headerBackVisible: false, gestureEnabled: false }}
          />
          <Stack.Screen name="MaleAvatar" component={MaleAvatarScreen} options={{ title: 'Your avatar' }} />
          <Stack.Screen name="FemaleAvatar" component={FemaleAvatarScreen} options={{ title: 'Your avatar' }} />
          <Stack.Screen
            name="FinalizedAvatar"
            component={FinalizedAvatarScreen}
            options={{ title: 'Finalized avatar' }}
          />
          <Stack.Screen name="Wardrobe" component={WardrobeScreen} options={{ title: 'Wardrobe' }} />
        </Stack.Navigator>
        <StatusBar style="dark" />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
