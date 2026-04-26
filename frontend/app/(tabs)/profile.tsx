import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Header } from "@/components/Header";
import { SIGNS } from "@/data/signs";
import { useAuthStore } from "@/store/authStore";
import { useSessionStore } from "@/store/sessionStore";
import { colors } from "@/theme/colors";

export default function ProfileScreen() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const signOut = useAuthStore((s) => s.signOut);
  const learnedSigns = useSessionStore((s) => s.learnedSigns);
  const history = useSessionStore((s) => s.history);

  const learned = learnedSigns.size;
  const total = SIGNS.length;

  return (
    <SafeAreaView edges={["top"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header />
      <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 40, gap: 16 }}>
        <Card padded={false} style={{ overflow: "hidden" }}>
          <LinearGradient
            colors={[`${colors.accent}33`, `${colors.brandPurple}33`]}
            style={{ padding: 24, flexDirection: "row", alignItems: "center", gap: 16 }}
          >
            <View
              style={{
                width: 68,
                height: 68,
                borderRadius: 34,
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: colors.bgSoft,
                borderWidth: 2,
                borderColor: colors.accent,
              }}
            >
              <Ionicons name="person" size={30} color={colors.accent} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={{ color: colors.text, fontSize: 22, fontWeight: "800" }}>
                {user?.name ?? "Listener"}
              </Text>
              <Text style={{ color: colors.textDim, fontSize: 13, marginTop: 2 }}>
                {user?.email ?? "—"}
              </Text>
            </View>
          </LinearGradient>
        </Card>

        <View style={{ flexDirection: "row", gap: 12 }}>
          <StatCard label="SIGNS LEARNED" value={`${learned}`} sub={`of ${total}`} tint={colors.accent} />
          <StatCard label="TRANSLATIONS" value={`${history.length}`} sub="this session" tint={colors.brandPurple} />
        </View>

        <Card>
          <Text style={{ color: colors.textDim, fontSize: 11, fontWeight: "700", letterSpacing: 1.5 }}>
            SETTINGS
          </Text>
          <SettingRow
            icon={<Ionicons name="language" size={18} color={colors.accent} />}
            label="App language"
            value="English"
          />
          <SettingRow
            icon={<Ionicons name="volume-high" size={18} color={colors.accent} />}
            label="TTS voice"
            value="Urdu (ur-PK)"
          />
          <SettingRow
            icon={<Ionicons name="options" size={18} color={colors.accent} />}
            label="Confidence threshold"
            value="70%"
            last
          />
        </Card>

        <Card>
          <Text style={{ color: colors.textDim, fontSize: 11, fontWeight: "700", letterSpacing: 1.5 }}>
            ABOUT
          </Text>
          <SettingRow
            icon={<Ionicons name="information-circle" size={18} color={colors.accent} />}
            label="Version"
            value="1.0.0"
          />
          <SettingRow
            icon={<Ionicons name="shield-checkmark" size={18} color={colors.accent} />}
            label="On-device inference"
            value="Enabled"
            last
          />
        </Card>

        <Button
          label="Sign out"
          variant="outline"
          icon={<Ionicons name="log-out-outline" size={18} color={colors.text} />}
          onPress={async () => {
            await signOut();
            router.replace("/auth/login");
          }}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

function StatCard({
  label,
  value,
  sub,
  tint,
}: {
  label: string;
  value: string;
  sub: string;
  tint: string;
}) {
  return (
    <View
      style={{
        flex: 1,
        backgroundColor: colors.bgCard,
        borderRadius: 18,
        borderWidth: 1,
        borderColor: colors.border,
        padding: 16,
      }}
    >
      <Text
        style={{
          color: colors.textDim,
          fontSize: 10,
          fontWeight: "700",
          letterSpacing: 1.4,
        }}
      >
        {label}
      </Text>
      <Text style={{ color: tint, fontSize: 28, fontWeight: "800", marginTop: 6 }}>
        {value}
      </Text>
      <Text style={{ color: colors.textDim, fontSize: 12, marginTop: 2 }}>{sub}</Text>
    </View>
  );
}

function SettingRow({
  icon,
  label,
  value,
  last,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  last?: boolean;
}) {
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        paddingVertical: 14,
        borderBottomWidth: last ? 0 : 1,
        borderBottomColor: colors.borderSoft,
      }}
    >
      <View
        style={{
          width: 34,
          height: 34,
          borderRadius: 10,
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: colors.bgSoft,
        }}
      >
        {icon}
      </View>
      <Text style={{ color: colors.text, fontSize: 14, marginLeft: 12, flex: 1 }}>
        {label}
      </Text>
      <Text style={{ color: colors.textDim, fontSize: 13 }}>{value}</Text>
    </View>
  );
}
