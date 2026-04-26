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
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Button } from "@/components/ui/Button";
import { useAuthStore } from "@/store/authStore";
import { colors } from "@/theme/colors";

export default function LoginScreen() {
  const router = useRouter();
  const signIn = useAuthStore((s) => s.signIn);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!email || !password) {
      Alert.alert("Missing fields", "Enter your email and password.");
      return;
    }
    setLoading(true);
    try {
      await signIn(email.trim(), password);
      router.replace("/(tabs)/translate");
    } catch (e) {
      Alert.alert("Sign in failed", (e as Error).message);
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
              colors={[`${colors.accent}55`, `${colors.brandPurple}33`]}
              style={{
                width: 88,
                height: 88,
                borderRadius: 44,
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <MaterialCommunityIcons name="hand-wave" size={44} color={colors.accent} />
            </LinearGradient>
            <Text
              style={{
                color: colors.text,
                fontSize: 28,
                fontWeight: "800",
                marginTop: 18,
              }}
            >
              Welcome back
            </Text>
            <Text style={{ color: colors.textDim, marginTop: 6 }}>
              Sign in to continue learning PSL.
            </Text>
          </View>

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
            placeholder="••••••••"
            secureTextEntry
          />

          <View style={{ marginTop: 24 }}>
            <Button label="Sign in" onPress={onSubmit} loading={loading} />
          </View>

          <Pressable
            onPress={() => router.push("/auth/signup")}
            style={{ marginTop: 22, alignItems: "center" }}
          >
            <Text style={{ color: colors.textDim }}>
              New here?{" "}
              <Text style={{ color: colors.accent, fontWeight: "700" }}>Create account</Text>
            </Text>
          </Pressable>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

export function InputField({
  label,
  icon,
  ...rest
}: {
  label: string;
  icon?: React.ReactNode;
} & React.ComponentProps<typeof TextInput>) {
  return (
    <View style={{ marginTop: 16 }}>
      <Text
        style={{
          color: colors.textDim,
          fontSize: 11,
          fontWeight: "700",
          letterSpacing: 1.3,
          marginBottom: 6,
        }}
      >
        {label.toUpperCase()}
      </Text>
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          gap: 10,
          paddingHorizontal: 14,
          backgroundColor: colors.bgCard,
          borderWidth: 1,
          borderColor: colors.border,
          borderRadius: 14,
        }}
      >
        {icon}
        <TextInput
          {...rest}
          placeholderTextColor={colors.textMuted}
          style={{
            flex: 1,
            color: colors.text,
            paddingVertical: 14,
            fontSize: 15,
          }}
        />
      </View>
    </View>
  );
}
