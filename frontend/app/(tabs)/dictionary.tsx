import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Card } from "@/components/ui/Card";
import { Header } from "@/components/Header";
import { SIGNS } from "@/data/signs";
import { colors } from "@/theme/colors";

// Dictionary page — intentionally minimal. The user asked to leave it
// empty for now; this placeholder keeps the tab slot occupied with a
// clean "coming soon" message + the overall sign count for continuity.
export default function DictionaryScreen() {
  return (
    <SafeAreaView edges={["top"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header />
      <View style={{ paddingHorizontal: 20, flex: 1 }}>
        <Text style={{ color: colors.text, fontSize: 32, fontWeight: "800" }}>
          Dictionary
        </Text>
        <Text
          style={{
            color: colors.textDim,
            fontSize: 12,
            letterSpacing: 1.5,
            fontWeight: "700",
            marginTop: 6,
          }}
        >
          {SIGNS.length} SIGNS SUPPORTED
        </Text>

        <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
          <Card padded={false} style={{ overflow: "hidden", width: "100%" }}>
            <LinearGradient
              colors={[`${colors.accent}22`, `${colors.brandPurple}22`]}
              style={{
                padding: 28,
                alignItems: "center",
                justifyContent: "center",
                gap: 14,
              }}
            >
              <View
                style={{
                  width: 68,
                  height: 68,
                  borderRadius: 34,
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: colors.bgSoft,
                  borderWidth: 1,
                  borderColor: colors.accent,
                }}
              >
                <MaterialCommunityIcons
                  name="book-open-variant"
                  size={30}
                  color={colors.accent}
                />
              </View>
              <Text
                style={{
                  color: colors.text,
                  fontSize: 22,
                  fontWeight: "800",
                  textAlign: "center",
                }}
              >
                Dictionary coming soon
              </Text>
              <Text
                style={{
                  color: colors.textDim,
                  fontSize: 14,
                  textAlign: "center",
                  lineHeight: 22,
                  paddingHorizontal: 6,
                }}
              >
                Searchable PSL dictionary with video tutorials, audio
                pronunciation and regional variants will live here.
              </Text>
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 6,
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 999,
                  backgroundColor: "rgba(0,0,0,0.35)",
                  borderWidth: 1,
                  borderColor: `${colors.accent}66`,
                }}
              >
                <Ionicons name="time" size={12} color={colors.accent} />
                <Text
                  style={{
                    color: colors.accent,
                    fontSize: 11,
                    fontWeight: "700",
                    letterSpacing: 1.2,
                  }}
                >
                  NEXT RELEASE
                </Text>
              </View>
            </LinearGradient>
          </Card>
        </View>
      </View>
    </SafeAreaView>
  );
}
