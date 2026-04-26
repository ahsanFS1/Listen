import { Ionicons } from "@expo/vector-icons";
import { useLocalSearchParams, useRouter } from "expo-router";
import { ScrollView, Text, View, Pressable, Linking } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Card } from "@/components/ui/Card";
import { Header } from "@/components/Header";
import { getCategoryById, getSignsByCategory } from "@/data/signs";
import { useTTS } from "@/hooks/useTTS";
import { useSessionStore } from "@/store/sessionStore";
import { colors } from "@/theme/colors";

export default function CategoryDetailScreen() {
  const { categoryId } = useLocalSearchParams<{ categoryId: string }>();
  const router = useRouter();
  const category = categoryId ? getCategoryById(categoryId) : undefined;
  const signs = categoryId ? getSignsByCategory(categoryId) : [];

  const { speak } = useTTS();
  const learnedSigns = useSessionStore((s) => s.learnedSigns);
  const markLearned = useSessionStore((s) => s.markLearned);
  const unmarkLearned = useSessionStore((s) => s.unmarkLearned);

  if (!category) {
    return (
      <SafeAreaView edges={["top"]} style={{ flex: 1, backgroundColor: colors.bg }}>
        <Header />
        <View style={{ padding: 20 }}>
          <Text style={{ color: colors.text }}>Category not found.</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView edges={["top", "bottom"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header />
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          paddingHorizontal: 20,
          paddingTop: 4,
        }}
      >
        <Pressable onPress={() => router.back()} hitSlop={10} style={{ paddingRight: 12 }}>
          <Ionicons name="chevron-back" size={26} color={colors.text} />
        </Pressable>
        <View>
          <Text style={{ color: colors.text, fontSize: 28, fontWeight: "800" }}>
            {category.name}
          </Text>
          <Text
            style={{
              color: colors.textDim,
              fontSize: 15,
              marginTop: 2,
              writingDirection: "rtl",
              textAlign: "right",
            }}
          >
            {category.nameUrdu}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20, paddingTop: 14, gap: 12 }}>
        <Text style={{ color: colors.textDim, fontSize: 12, fontWeight: "700", letterSpacing: 1.5 }}>
          {signs.length} SIGNS
        </Text>

        {signs.map((s) => {
          const learned = learnedSigns.has(s.id);
          return (
            <Card key={s.id}>
              <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: colors.text, fontSize: 17, fontWeight: "700" }}>
                    {s.english}
                  </Text>
                  <Text
                    style={{
                      color: colors.accent,
                      fontSize: 20,
                      marginTop: 2,
                      writingDirection: "rtl",
                    }}
                  >
                    {s.urdu}
                  </Text>
                  {s.note ? (
                    <Text
                      style={{
                        color: colors.textDim,
                        fontSize: 12,
                        marginTop: 6,
                        fontStyle: "italic",
                      }}
                    >
                      Tip: {s.note}
                    </Text>
                  ) : null}
                </View>

                <Pressable
                  onPress={() => speak(s.urdu)}
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: 20,
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: `${colors.accent}22`,
                    borderWidth: 1,
                    borderColor: `${colors.accent}66`,
                  }}
                >
                  <Ionicons name="volume-high" size={18} color={colors.accent} />
                </Pressable>
              </View>

              <View style={{ flexDirection: "row", gap: 8, marginTop: 14 }}>
                {s.url ? (
                  <Pressable
                    onPress={() => Linking.openURL(s.url!)}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      paddingHorizontal: 12,
                      paddingVertical: 8,
                      borderRadius: 10,
                      backgroundColor: colors.bgSoft,
                      borderWidth: 1,
                      borderColor: colors.border,
                    }}
                  >
                    <Ionicons name="play-circle" size={16} color={colors.text} />
                    <Text style={{ color: colors.text, fontSize: 12, fontWeight: "600" }}>
                      Watch video
                    </Text>
                  </Pressable>
                ) : null}
                <Pressable
                  onPress={() =>
                    learned ? unmarkLearned(s.id) : markLearned(s.id)
                  }
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 6,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                    borderRadius: 10,
                    backgroundColor: learned ? `${colors.ok}20` : colors.bgSoft,
                    borderWidth: 1,
                    borderColor: learned ? `${colors.ok}66` : colors.border,
                  }}
                >
                  <Ionicons
                    name={learned ? "checkmark-circle" : "ellipse-outline"}
                    size={16}
                    color={learned ? colors.ok : colors.textDim}
                  />
                  <Text
                    style={{
                      color: learned ? colors.ok : colors.text,
                      fontSize: 12,
                      fontWeight: "600",
                    }}
                  >
                    {learned ? "Learned" : "Mark learned"}
                  </Text>
                </Pressable>
              </View>
            </Card>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}
