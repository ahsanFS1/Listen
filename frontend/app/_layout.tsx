import "../global.css";
import { Stack, useRouter, useSegments } from "expo-router";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { useEffect } from "react";
import { View, ActivityIndicator } from "react-native";

import { colors } from "@/theme/colors";
import { useAuthStore } from "@/store/authStore";
import { useSessionStore } from "@/store/sessionStore";

export default function RootLayout() {
  const hydrate = useAuthStore((s) => s.hydrate);
  const hydrateSession = useSessionStore((s) => s.hydrate);
  const loading = useAuthStore((s) => s.loading);
  const onboarded = useAuthStore((s) => s.onboarded);
  const user = useAuthStore((s) => s.user);

  const router = useRouter();
  const segments = useSegments();

  useEffect(() => {
    hydrate();
    hydrateSession();
  }, [hydrate, hydrateSession]);

  useEffect(() => {
    if (loading) return;
    const seg0 = segments[0] as string | undefined;

    if (!onboarded) {
      if (seg0 !== "onboarding") router.replace("/onboarding");
      return;
    }
    if (!user) {
      if (seg0 !== "auth" && seg0 !== "onboarding") router.replace("/auth/login");
      return;
    }
    if (seg0 === "auth" || seg0 === "onboarding") {
      router.replace("/(tabs)/translate");
    }
  }, [loading, onboarded, user, segments, router]);

  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: colors.bg }}>
      <SafeAreaProvider>
        <StatusBar style="light" />
        {loading ? (
          <View
            style={{
              flex: 1,
              backgroundColor: colors.bg,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ActivityIndicator color={colors.accent} />
          </View>
        ) : (
          <Stack
            screenOptions={{
              headerShown: false,
              contentStyle: { backgroundColor: colors.bg },
              animation: "fade",
            }}
          />
        )}
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
