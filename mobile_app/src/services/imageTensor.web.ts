import * as tf from '@tensorflow/tfjs';

/** A decoded image, ready for on-device analysis. */
export interface LoadedImage {
  tensor: tf.Tensor3D;
  /** Data URI of the (possibly resized) image, e.g. for cropping a texture from it. */
  uri: string;
  width: number;
  height: number;
}

/**
 * Decodes `uri` into an RGB tensor, resizing it to `maxWidth` first to keep
 * on-device inference fast. Callers must `dispose()` the returned tensor.
 */
export async function loadImageAsTensor(uri: string, maxWidth = 640): Promise<LoadedImage> {
  const image = await loadHtmlImage(uri);
  const scale = Math.min(1, maxWidth / image.width);
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get a 2D canvas context.');
  }
  ctx.drawImage(image, 0, 0, width, height);

  const tensor = tf.browser.fromPixels(canvas) as tf.Tensor3D;
  return { tensor, uri: canvas.toDataURL('image/jpeg', 0.9), width, height };
}

/**
 * Decodes `uri` (already a JPEG/PNG/data URI) into an RGB tensor without
 * resizing. Callers must `dispose()` the returned tensor.
 */
export async function decodeImageUri(uri: string): Promise<{ tensor: tf.Tensor3D; width: number; height: number }> {
  const image = await loadHtmlImage(uri);
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get a 2D canvas context.');
  }
  ctx.drawImage(image, 0, 0);

  const tensor = tf.browser.fromPixels(canvas) as tf.Tensor3D;
  return { tensor, width: image.width, height: image.height };
}

function loadHtmlImage(uri: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Failed to load image.'));
    image.src = uri;
  });
}
