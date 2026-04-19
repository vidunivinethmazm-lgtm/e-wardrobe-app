# eWardrobeAI — Expo Mobile App

## Setup

```bash
cd ewardrobe-app
npm install
```

## Run

```bash
# Start backend first (in vineths_wardrobe folder)
python mobile_app.py

# Then start Expo
npx expo start
```

## Physical Device (important)
Edit `src/api/client.ts` and replace `API_BASE` with your PC's local IP:
```ts
export const API_BASE = 'http://192.168.X.X:8000';
```
Find your IP: run `ipconfig` in CMD and look for IPv4 Address.

## Android Emulator
Keep `API_BASE = 'http://10.0.2.2:8000'` (routes to host machine).
