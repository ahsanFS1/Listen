import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { useState } from "react";
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Button } from "@/components/ui/Button";
import { useAuthStore } from "@/store/authStore";
import { colors } from "@/theme/colors";
import { InputField } from "./login";

export default function SignUpScreen() {
  const router = useRouter();
  const signUp = useAuthStore((s) => s.signUp);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!name || !email || !password) {
      Alert.alert("Missing fields", "Fill in all fields to continue.");
      return;
    }
    setLoading(true);
    try {
      await signUp(name.trim(), email.trim(), password);
      router.replace("/(tabs)/translate");
    } catch (e) {
      Alert.alert("Sign up failed", (e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView edges={["top", "bottom"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={{ flex: 1 }}
      >
        <ScrollView contentContainerStyle={{ padding: 24, paddingTop: 40, flexGrow: 1 }}>
          <View style={{ alignItems: "center", marginBottom: 32 }}>
            <LinearGradient
              colors={[`${colors.brandPurple}55`, `${colors.accent}33`]}
              style={{
                width: 88,
                height: 88,
                borderRadius: 44,
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <MaterialCommunityIcons name="account-plus" size={40} color={colors.accent} />
            </LinearGradient>
            <Text
              style={{
                color: colors.text,
                fontSize: 28,
                fontWeight: "800",
                marginTop: 18,
              }}
            >
              Create account
            </Text>
            <Text style={{ color: colors.textDim, marginTop: 6, textAlign: "center" }}>
              Join the PSL learning community.
            </Text>
          </View>

          <InputField
            label="Name"
            icon={<Ionicons name="person" size={18} color={colors.textDim} />}
            value={name}
            onChangeText={setName}
            placeholder="Your name"
          />
          <InputField
            label="Email"
            icon={<Ionicons name="mail" size={18} color={colors.textDim} />}
            value={email}
            onChangeText={setEmail}
            placeholder="you@example.com"
            autoCapitalize="none"
            keyboardType="email-address"
          />
          <InputField
            label="Password"
            icon={<Ionicons name="lock-closed" size={18} color={colors.textDim} />}
            value={password}
            onChangeText={setPassword}
            placeholder="Choose a password"
            secureTextEntry
          />

          <View style={{ marginTop: 24 }}>
            <Button label="Create account" onPress={onSubmit} loading={loading} />
          </View>

          <Pressable
            onPress={() => router.replace("/auth/login")}
            style={{ marginTop: 22, alignItems: "center" }}
          >
            <Text style={{ color: colors.textDim }}>
              Already have an account?{" "}
              <Text style={{ color: colors.accent, fontWeight: "700" }}>Sign in</Text>
            </Text>
          </Pressable>

          <Text
            style={{
              color: colors.textMuted,
              fontSize: 11,
              textAlign: "center",
              marginTop: 18,
              lineHeight: 16,
            }}
          >
            Accounts are stored locally on this device. Swap to Firebase or
            Supabase before production launch.
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
