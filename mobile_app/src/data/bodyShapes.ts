import type { BodyShape } from '../types';

export const BODY_SHAPE_INFO: Record<BodyShape, { title: string; description: string }> = {
  Hourglass: {
    title: 'Hourglass',
    description: 'Balanced bust and hips with a defined waist. Fitted silhouettes show off your shape.',
  },
  Pear: {
    title: 'Pear',
    description: 'Hips are wider than the bust. A-line skirts and structured tops balance your frame.',
  },
  Apple: {
    title: 'Apple',
    description: 'Fuller through the middle with a less defined waist. Empire waists and flowy fabrics flatter.',
  },
  Rectangle: {
    title: 'Rectangle',
    description: 'Bust, waist, and hips are similarly proportioned. Belts and layers add definition.',
  },
  InvertedTriangle: {
    title: 'Inverted Triangle',
    description: 'Shoulders and bust are wider than the hips. Wide-leg bottoms balance your upper body.',
  },
};
