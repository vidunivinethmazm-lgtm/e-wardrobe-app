import React, { useState } from 'react';
import { View, StyleSheet, StatusBar, Alert } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import StepNav from './src/components/StepNav';
import SelfieScreen from './src/screens/SelfieScreen';
import MeasurementsScreen, { Measurements } from './src/screens/MeasurementsScreen';
import ProcessingScreen from './src/screens/ProcessingScreen';
import TryOnScreen from './src/screens/TryOnScreen';

import { callTryOn } from './src/api/client';
import { colors } from './src/constants/theme';

type Step = 1 | 2 | 3 | 4;

export default function App() {
  const [step,       setStep]       = useState<Step>(1);
  const [selfieUri,  setSelfieUri]  = useState<string | null>(null);
  const [apiPromise, setApiPromise] = useState<Promise<any> | null>(null);
  const [result,     setResult]     = useState<any>(null);
  const [measures,   setMeasures]   = useState<Measurements | null>(null);

  function handleSelfieReady(uri: string) {
    setSelfieUri(uri);
  }

  function handleMeasurementsDone(m: Measurements) {
    if (!selfieUri) { Alert.alert('Please capture a selfie first'); return; }
    setMeasures(m);

    const promise = callTryOn({
      selfie: { uri: selfieUri, name: 'selfie.jpg', type: 'image/jpeg' },
      shoulder_width_cm: m.shoulder,
      chest_cm:          m.chest,
      waist_cm:          m.waist,
      height_cm:         m.height,
      hip_cm:            m.hip || undefined,
      inseam_cm:         m.inseam || undefined,
      styles:            m.styles,
      occasion:          m.occasion,
      top_k:             5,
    });

    setApiPromise(promise);
    setStep(3);
  }

  function handleProcessingDone(r: any) {
    setResult(r);
    setStep(4);
  }

  function handleProcessingError(msg: string) {
    Alert.alert('Processing Failed', msg, [
      { text: 'Try Again', onPress: () => setStep(2) },
    ]);
  }

  function handleReset() {
    setSelfieUri(null);
    setApiPromise(null);
    setResult(null);
    setMeasures(null);
    setStep(1);
  }

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor={colors.bg} />
      <SafeAreaView style={styles.safe}>
        {step !== 3 && <StepNav current={step} />}

        <View style={styles.body}>
          {step === 1 && (
            <SelfieScreen
              onSelfieReady={handleSelfieReady}
              onNext={() => setStep(2)}
            />
          )}
          {step === 2 && (
            <MeasurementsScreen
              onBack={() => setStep(1)}
              onNext={handleMeasurementsDone}
            />
          )}
          {step === 3 && (
            <ProcessingScreen
              apiPromise={apiPromise}
              onDone={handleProcessingDone}
              onError={handleProcessingError}
            />
          )}
          {step === 4 && result && (
            <TryOnScreen
              result={result}
              selfieUri={selfieUri}
              onReset={handleReset}
            />
          )}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  body: { flex: 1 },
});
