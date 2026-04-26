import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { useMemo, useState } from "react";
import {
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { ProgressRing } from "@/components/ui/ProgressRing";
import { Header } from "@/components/Header";
import { CATEGORIES, SIGNS, getSignsByCategory } from "@/data/signs";
import { useSessionStore } from "@/store/sessionStore";
import { colors } from "@/theme/colors";

export default function LearnScreen() {
  const router = useRouter();
  const learnedSigns = useSessionStore((s) => s.learnedSigns);
  const [query, setQuery] = useState("");

  const totalSigns = SIGNS.length;
  const learnedCount = useMemo(
    () => SIGNS.filter((s) => learnedSigns.has(s.id)).length,
    [learnedSigns],
  );

  const overall = learnedCount / Math.max(1, totalSigns);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return CATEGORIES;
    return CATEGORIES.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.nameUrdu.includes(q) ||
        getSignsByCategory(c.id).some((s) => s.english.toLowerCase().includes(q)),
    );
  }, [query]);

  return (
    <SafeAreaView edges={["top"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header />
      <ScrollView contentContainerStyle={{ paddingBottom: 40 }}>
        <View style={{ paddingHorizontal: 20, marginTop: 6 }}>
          <Text style={{ color: colors.text, fontSize: 32, fontWeight: "800" }}>
            Master PSL
          </Text>
          <Text
            style={{
              color: colors.textDim,
              fontSize: 14,
              marginTop: 6,
              lineHeight: 20,
            }}
          >
            Explore {totalSigns} signs across core categories to enhance your
            Pakistani Sign Language vocabulary.
          </Text>

          {/* Search */}
          <View
            style={{
              marginTop: 16,
              flexDirection: "row",
              alignItems: "center",
              backgroundColor: colors.bgCard,
              borderRadius: 14,
              borderWidth: 1,
              borderColor: colors.border,
              paddingHorizontal: 14,
            }}
          >
            <Ionicons name="search" size={18} color={colors.textDim} />
            <TextInput
              placeholder="Search signs or categories..."
              placeholderTextColor={colors.textMuted}
              value={query}
              onChangeText={setQuery}
              style={{
                flex: 1,
                color: colors.text,
                paddingVertical: 14,
                paddingHorizontal: 10,
                fontSize: 14,
              }}
            />
          </View>

          {/* Overall progress */}
          <Card style={{ marginTop: 16 }}>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <View>
                <Text
                  style={{
                    color: colors.textDim,
                    fontSize: 11,
                    fontWeight: "700",
                    letterSpacing: 1.5,
                  }}
                >
                  OVERALL PROGRESS
                </Text>
                <Text
                  style={{
                    color: colors.text,
                    fontSize: 28,
                    fontWeight: "800",
                    marginTop: 4,
                  }}
                >
                  {learnedCount}{" "}
                  <Text style={{ color: colors.textDim, fontSize: 18 }}>
                    / {totalSigns}{" "}
                  </Text>
                  <Text style={{ color: colors.textDim, fontSize: 16 }}>Signs</Text>
                </Text>
              </View>
              <ProgressRing
                value={overall}
                size={74}
                stroke={8}
                label={`${Math.round(overall * 100)}%`}
              />
            </View>
          </Card>
        </View>

        {/* Category list */}
        <View style={{ paddingHorizontal: 20, marginTop: 18, gap: 14 }}>
          {filtered.map((c) => {
            const signsInCat = getSignsByCategory(c.id);
            const total = signsInCat.length;
            const done = signsInCat.filter((s) => learnedSigns.has(s.id)).length;
            const pct = total > 0 ? done / total : 0;
            const doneLabel =
              done === 0
                ? "Not Started"
                : pct === 1
                ? "Completed"
                : `${Math.round(pct * 100)}% Done`;

            return (
              <Card
                key={c.id}
                onPress={() => router.push(`/learn/${c.id}`)}
                glow
              >
                <View style={{ flexDirection: "row", alignItems: "center" }}>
                  <View
                    style={{
                      width: 46,
                      height: 46,
                      borderRadius: 12,
                      alignItems: "center",
                      justifyContent: "center",
                      backgroundColor: `${c.gradient[0]}25`,
                      borderWidth: 1,
                      borderColor: `${c.gradient[0]}40`,
                    }}
                  >
                    <Text style={{ fontSize: 22 }}>{c.icon}</Text>
                  </View>
                  <View style={{ flex: 1, marginLeft: 12 }}>
                    <View
                      style={{
                        flexDirection: "row",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Text style={{ color: colors.text, fontSize: 17, fontWeight: "700" }}>
                        {c.name}
                      </Text>
                      <View
                        style={{
                          backgroundColor: colors.bgSoft,
                          paddingHorizontal: 10,
                          paddingVertical: 4,
                          borderRadius: 999,
                        }}
                      >
                        <Text
                          style={{
                            color: colors.textDim,
                            fontSize: 11,
                            fontWeight: "700",
                          }}
                        >
                          {done}/{total}
                        </Text>
                      </View>
                    </View>
                    <Text
                      style={{
                        color: colors.textDim,
                        fontSize: 15,
                        marginTop: 2,
                        writingDirection: "rtl",
                        textAlign: "right",
                      }}
                    >
                      {c.nameUrdu}
                    </Text>
                  </View>
                </View>

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginTop: 12,
                    gap: 10,
                  }}
                >
                  <View style={{ flex: 1 }}>
                    <ProgressBar value={pct} gradient={c.gradient} />
                  </View>
                  <Text
                    style={{
                      color:
                        done === 0
                          ? colors.textDim
                          : pct === 1
                          ? colors.ok
                          : colors.accent,
                      fontSize: 11,
                      fontWeight: "700",
                    }}
                  >
                    {doneLabel}
                  </Text>
                </View>
              </Card>
            );
          })}

          {/* Daily challenge */}
          <Card padded={false} style={{ overflow: "hidden" }}>
            <LinearGradient
              colors={[`${colors.brandPurple}55`, `${colors.accent}55`]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{ padding: 18 }}
            >
              <Text
                style={{
                  color: colors.text,
                  fontSize: 11,
                  fontWeight: "800",
                  letterSpacing: 1.5,
                }}
              >
                DAILY CHALLENGE
              </Text>
              <Text
                style={{
                  color: colors.text,
                  fontSize: 22,
                  fontWeight: "800",
                  marginTop: 8,
                }}
              >
                Master &quot;Thank You&quot;
              </Text>
              <Text style={{ color: colors.text, fontSize: 13, marginTop: 6, opacity: 0.9 }}>
                Earn double XP today by mastering courtesy signs.
              </Text>
              <Pressable
                onPress={() => router.push("/learn/greetings")}
                style={{
                  alignSelf: "flex-start",
                  marginTop: 14,
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 6,
                  paddingHorizontal: 14,
                  paddingVertical: 8,
                  borderRadius: 999,
                  backgroundColor: "rgba(0,0,0,0.4)",
                }}
              >
                <Ionicons name="play" size={12} color={colors.accent} />
                <Text style={{ color: colors.accent, fontWeight: "700", fontSize: 12 }}>
                  Start
                </Text>
              </Pressable>
            </LinearGradient>
          </Card>
        </View>
      </ScrollView>

      {/* FAB */}
      <Pressable
        onPress={() => router.push("/(tabs)/translate")}
        style={{
          position: "absolute",
          right: 20,
          bottom: 92,
          width: 56,
          height: 56,
          borderRadius: 28,
          backgroundColor: colors.accent,
          alignItems: "center",
          justifyContent: "center",
          shadowColor: colors.accent,
          shadowOpacity: 0.4,
          shadowRadius: 14,
          shadowOffset: { width: 0, height: 4 },
        }}
      >
        <Ionicons name="add" size={28} color="#0B1020" />
      </Pressable>
    </SafeAreaView>
  );
}
