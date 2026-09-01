import { Platform } from 'react-native';

export const colors = {
  background: '#F6F4FB',
  surface: '#FFFFFF',
  primary: '#6C5CE7',
  primaryDark: '#5645D9',
  accent: '#FF7AA2',
  text: '#241F33',
  textMuted: '#8A8499',
  border: '#ECE9F6',
  success: '#3FBF8F',
  danger: '#E5586C',
};

export const gradients = {
  primary: [colors.primary, '#9B6CF6'] as const,
  accent: [colors.accent, '#FFB199'] as const,
  hero: ['#6C5CE7', '#A78BFA', '#FFB199'] as const,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const radii = {
  sm: 8,
  md: 16,
  lg: 24,
  pill: 999,
};

export const typography = {
  title: {
    fontSize: 28,
    fontWeight: '800' as const,
    color: colors.text,
  },
  subtitle: {
    fontSize: 16,
    fontWeight: '500' as const,
    color: colors.textMuted,
  },
  heading: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: colors.text,
  },
  body: {
    fontSize: 15,
    fontWeight: '400' as const,
    color: colors.text,
  },
  label: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: colors.textMuted,
    textTransform: 'uppercase' as const,
    letterSpacing: 0.5,
  },
  button: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: colors.surface,
  },
};

export const shadow = Platform.select({
  android: {
    elevation: 4,
  },
  default: {
    shadowColor: '#2A1F4D',
    shadowOpacity: 0.12,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
  },
});
