import * as ImageManipulator from 'expo-image-manipulator';

import type { CapturePose } from '../types';

/**
 * Resizes/compresses the captured photo. Web has no filesystem to write
 * `front.jpg`/`left.jpg`/`right.jpg` to, so this returns a data URI for
 * on-device preview/analysis instead - `pose` is accepted for API symmetry
 * with the native implementation.
 */
export async function saveCapture(_pose: CapturePose, sourceUri: string): Promise<string> {
  const resized = await ImageManipulator.manipulateAsync(sourceUri, [{ resize: { width: 1024 } }], {
    format: ImageManipulator.SaveFormat.JPEG,
    compress: 0.8,
  });
  return resized.uri;
}
