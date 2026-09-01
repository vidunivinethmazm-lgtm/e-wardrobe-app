import { Directory, File, Paths } from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';

import type { CapturePose } from '../types';

const CAPTURE_DIR_NAME = 'avatar_capture';

/**
 * Resizes/compresses a captured photo and writes it to
 * `<documents>/avatar_capture/<pose>.jpg`, returning its local `file://` URI
 * for on-device preview/analysis. This URI is never sent to the server or
 * returned as the final avatar URL.
 */
export async function saveCapture(pose: CapturePose, sourceUri: string): Promise<string> {
  const resized = await ImageManipulator.manipulateAsync(sourceUri, [{ resize: { width: 1024 } }], {
    format: ImageManipulator.SaveFormat.JPEG,
    compress: 0.8,
    base64: true,
  });
  if (!resized.base64) {
    throw new Error('Failed to encode captured photo.');
  }

  const directory = new Directory(Paths.document, CAPTURE_DIR_NAME);
  directory.create({ intermediates: true, idempotent: true });

  const file = new File(directory, `${pose}.jpg`);
  file.create({ overwrite: true });
  file.write(resized.base64, { encoding: 'base64' });
  return file.uri;
}
