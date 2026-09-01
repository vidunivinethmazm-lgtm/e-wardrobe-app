import '@tensorflow/tfjs-react-native';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg, fetch as tfFetch } from '@tensorflow/tfjs-react-native';
import * as ImageManipulator from 'expo-image-manipulator';

/** A decoded image, ready for on-device analysis. */
export interface LoadedImage {
  tensor: tf.Tensor3D;
  /** Local URI of the (possibly resized) image, e.g. for cropping a texture from it. */
  uri: string;
  width: number;
  height: number;
}

/**
 * Decodes `uri` (already a JPEG/PNG file) into an RGB tensor without
 * resizing. Callers must `dispose()` the returned tensor.
 */
export async function decodeImageUri(uri: string): Promise<{ tensor: tf.Tensor3D; width: number; height: number }> {
  await tf.ready();
  const response = await tfFetch(uri, {}, { isBinary: true });
  const buffer = await response.arrayBuffer();
  const tensor = decodeJpeg(new Uint8Array(buffer)) as tf.Tensor3D;
  const [height, width] = tensor.shape;
  return { tensor, width, height };
}

/**
 * Decodes `uri` into an RGB tensor, resizing it to `maxWidth` first to keep
 * on-device inference fast. Callers must `dispose()` the returned tensor.
 */
export async function loadImageAsTensor(uri: string, maxWidth = 640): Promise<LoadedImage> {
  const resized = await ImageManipulator.manipulateAsync(uri, [{ resize: { width: maxWidth } }], {
    format: ImageManipulator.SaveFormat.JPEG,
    compress: 0.9,
  });

  const { tensor, width, height } = await decodeImageUri(resized.uri);
  return { tensor, uri: resized.uri, width, height };
}
