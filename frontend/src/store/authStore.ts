import AsyncStorage from "@react-native-async-storage/async-storage";
import { create } from "zustand";

// Local mock auth — backed by AsyncStorage.
// In production: swap this with Firebase Auth / Supabase.
// (MongoDB would require a backend, which conflicts with the on-device-only
// architecture the user chose.)

export type User = {
  id: string;
  name: string;
  email: string;
  createdAt: number;
};

const USER_KEY = "listen.user";
const ONBOARDED_KEY = "listen.onboarded";

type AuthState = {
  user: User | null;
  onboarded: boolean;
  loading: boolean;
  hydrate: () => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<User>;
  signIn: (email: string, password: string) => Promise<User>;
  signOut: () => Promise<void>;
  completeOnboarding: () => Promise<void>;
};

const fakeDelay = (ms = 400) => new Promise((r) => setTimeout(r, ms));

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  onboarded: false,
  loading: true,

  hydrate: async () => {
    try {
      const [userJson, onboardedFlag] = await Promise.all([
        AsyncStorage.getItem(USER_KEY),
        AsyncStorage.getItem(ONBOARDED_KEY),
      ]);
      set({
        user: userJson ? (JSON.parse(userJson) as User) : null,
        onboarded: onboardedFlag === "true",
        loading: false,
      });
    } catch {
      set({ loading: false });
    }
  },

  signUp: async (name, email, _password) => {
    await fakeDelay();
    const user: User = {
      id: `u_${Date.now()}`,
      name,
      email,
      createdAt: Date.now(),
    };
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(user));
    set({ user });
    return user;
  },

  signIn: async (email, _password) => {
    await fakeDelay();
    // Mock: accept anything, remember the email.
    const existing = await AsyncStorage.getItem(USER_KEY);
    const user: User = existing
      ? JSON.parse(existing)
      : {
          id: `u_${Date.now()}`,
          name: email.split("@")[0] || "Listener",
          email,
          createdAt: Date.now(),
        };
    if (user.email !== email) {
      user.email = email;
      user.name = email.split("@")[0] || user.name;
    }
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(user));
    set({ user });
    return user;
  },

  signOut: async () => {
    await AsyncStorage.removeItem(USER_KEY);
    set({ user: null });
  },

  completeOnboarding: async () => {
    await AsyncStorage.setItem(ONBOARDED_KEY, "true");
    set({ onboarded: true });
  },
}));
