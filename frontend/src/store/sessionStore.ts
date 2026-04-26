import AsyncStorage from "@react-native-async-storage/async-storage";
import { create } from "zustand";

// Tracks the user's live translation session: committed words + history.
// Also tracks per-category learning progress for the Learn page.

type CommittedWord = {
  label: string;
  english: string;
  urdu: string;
  timestamp: number;
};

type SessionState = {
  history: CommittedWord[];
  learnedSigns: Set<string>; // sign ids the user has marked complete
  confidenceThreshold: number;

  commitWord: (w: Omit<CommittedWord, "timestamp">) => void;
  clearHistory: () => void;
  undoLast: () => void;
  markLearned: (signId: string) => Promise<void>;
  unmarkLearned: (signId: string) => Promise<void>;
  hydrate: () => Promise<void>;
  setConfidenceThreshold: (v: number) => void;
};

const LEARNED_KEY = "listen.learnedSigns";

export const useSessionStore = create<SessionState>((set, get) => ({
  history: [],
  learnedSigns: new Set(),
  confidenceThreshold: 0.7,

  commitWord: (w) =>
    set((s) => ({
      history: [...s.history, { ...w, timestamp: Date.now() }],
    })),

  clearHistory: () => set({ history: [] }),

  undoLast: () =>
    set((s) => ({ history: s.history.slice(0, -1) })),

  markLearned: async (signId) => {
    const next = new Set(get().learnedSigns);
    next.add(signId);
    set({ learnedSigns: next });
    await AsyncStorage.setItem(LEARNED_KEY, JSON.stringify([...next]));
  },

  unmarkLearned: async (signId) => {
    const next = new Set(get().learnedSigns);
    next.delete(signId);
    set({ learnedSigns: next });
    await AsyncStorage.setItem(LEARNED_KEY, JSON.stringify([...next]));
  },

  hydrate: async () => {
    try {
      const raw = await AsyncStorage.getItem(LEARNED_KEY);
      if (raw) {
        const arr = JSON.parse(raw) as string[];
        set({ learnedSigns: new Set(arr) });
      }
    } catch {
      /* noop */
    }
  },

  setConfidenceThreshold: (v) => set({ confidenceThreshold: v }),
}));

export const sentenceFromHistory = (h: CommittedWord[]): string =>
  h.map((w) => w.urdu).join(" ");

export const englishSentenceFromHistory = (h: CommittedWord[]): string =>
  h.map((w) => w.english).join(" ");
