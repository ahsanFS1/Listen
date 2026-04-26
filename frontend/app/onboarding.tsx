import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { Text, View, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Button } from "@/components/ui/Button";
import { Header } from "@/components/Header";
import { colors } from "@/theme/colors";
import { useAuthStore } from "@/store/authStore";

export default function OnboardingScreen() {
  const router = useRouter();
  const completeOnboarding = useAuthStore((s) => s.completeOnboarding);
  const user = useAuthStore((s) => s.user);

  const onGetStarted = async () => {
    await completeOnboarding();
    if (user) router.replace("/(tabs)/translate");
    else router.replace("/auth/signup");
  };

  return (
    <SafeAreaView edges={["top", "bottom"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header showMenu={false} showLang showAvatar />

      <ScrollView contentContainerStyle={{ paddingBottom: 40 }}>
        <View className="items-center mt-6 mb-5">
          <View
            style={{
              width: 220,
              height: 220,
              borderRadius: 110,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {/* Outer glow ring */}
            <View
              style={{
                position: "absolute",
                width: 220,
                height: 220,
                borderRadius: 110,
                borderWidth: 1.5,
                borderColor: colors.accent,
                opacity: 0.35,
              }}
            />
            <View
              style={{
                position: "absolute",
                width: 180,
                height: 180,
                borderRadius: 90,
                borderWidth: 1,
                borderColor: colors.accent,
                opacity: 0.2,
              }}
            />
            <LinearGradient
              colors={[`${colors.accent}55`, `${colors.brandPurple}22`, "transparent"]}
              style={{
                position: "absolute",
                width: 200,
                height: 200,
                borderRadius: 100,
              }}
            />
            <MaterialCommunityIcons
              name="hand-wave"
              size={96}
              color={colors.accent}
              style={{
                textShadowColor: colors.accent,
                textShadowRadius: 20,
              }}
            />
          </View>
          <View
            style={{
              marginTop: -12,
              paddingHorizontal: 16,
              paddingVertical: 7,
              borderRadius: 999,
              backgroundColor: colors.bgSoft,
              borderWidth: 1,
              borderColor: colors.accent,
              flexDirection: "row",
              alignItems: "center",
              gap: 6,
            }}
          >
            <View
              style={{
                width: 6,
                height: 6,
                borderRadius: 3,
                backgroundColor: colors.accent,
              }}
            />
            <Text style={{ color: colors.accent, fontWeight: "700", fontSize: 12, letterSpacing: 1 }}>
              AI READY
            </Text>
          </View>
        </View>

        <View className="px-6">
          <Text
            style={{
              color: colors.text,
              fontSize: 34,
              fontWeight: "800",
              textAlign: "center",
              lineHeight: 42,
            }}
          >
            Bridging the communication gap,{" "}
            <Text style={{ color: colors.accent }}>one sign at a time.</Text>
          </Text>

          <Text
            style={{
              color: colors.textDim,
              fontSize: 18,
              textAlign: "center",
              marginTop: 18,
              lineHeight: 30,
              writingDirection: "rtl",
            }}
          >
            رابطوں کی دوری ختم کریں، اشاروں کی زبان کے ساتھ۔
          </Text>
        </View>

        <View className="px-6 mt-8 gap-3">
          <FeatureRow
            icon={<Ionicons name="videocam" size={22} color={colors.accent} />}
            title="Live Vision"
            subtitle="Real-time PSL detection"
            tint={colors.accent}
          />
          <FeatureRow
            icon={<Ionicons name="school" size={22} color={colors.brandPurple} />}
            title="Guided Path"
            subtitle="Learn as you communicate"
            tint={colors.brandPurple}
          />
        </View>

        <View className="px-6 mt-8">
          <Button
            label="Get Started"
            onPress={onGetStarted}
            trailingIcon={<Ionicons name="arrow-forward" size={18} color="#0B1020" />}
          />
          <Text
            style={{
              color: colors.textMuted,
              textAlign: "center",
              marginTop: 14,
              fontSize: 12,
            }}
          >
            Empowering the Deaf Community in Pakistan
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function FeatureRow({
  icon,
  title,
  subtitle,
  tint,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  tint: string;
}) {
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        gap: 14,
        padding: 14,
        borderRadius: 16,
        backgroundColor: colors.bgCard,
        borderWidth: 1,
        borderColor: colors.border,
      }}
    >
      <View
        style={{
          width: 46,
          height: 46,
          borderRadius: 12,
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: `${tint}20`,
          borderWidth: 1,
          borderColor: `${tint}40`,
        }}
      >
        {icon}
      </View>
      <View>
        <Text style={{ color: colors.text, fontWeight: "700", fontSize: 16 }}>
          {title}
        </Text>
        <Text style={{ color: colors.textDim, fontSize: 13, marginTop: 2 }}>
          {subtitle}
        </Text>
      </View>
    </View>
  );
}
