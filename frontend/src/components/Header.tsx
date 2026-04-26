import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { Pressable, Text, View } from "react-native";
import { colors } from "@/theme/colors";

type Props = {
  showMenu?: boolean;
  showAvatar?: boolean;
  showLang?: boolean;
  trailing?: React.ReactNode;
};

export function Header({
  showMenu = true,
  showAvatar = true,
  showLang = false,
  trailing,
}: Props) {
  const router = useRouter();
  return (
    <View className="flex-row items-center justify-between px-5 pt-3 pb-3">
      <View className="flex-row items-center gap-3">
        {showMenu ? (
          <Pressable hitSlop={10} className="p-1">
            <Ionicons name="menu" size={24} color={colors.text} />
          </Pressable>
        ) : null}
        <Text
          className="text-accent"
          style={{
            fontSize: 18,
            fontWeight: "800",
            letterSpacing: 3,
            color: colors.accent,
          }}
        >
          LISTEN
        </Text>
      </View>

      <View className="flex-row items-center gap-3">
        {trailing}
        {showLang ? (
          <Text className="text-text-dim text-sm" style={{ color: colors.textDim }}>
            EN | اردو
          </Text>
        ) : null}
        {showAvatar ? (
          <Pressable
            onPress={() => router.push("/(tabs)/profile")}
            hitSlop={8}
            className="w-9 h-9 rounded-full items-center justify-center"
            style={{
              backgroundColor: colors.bgSoft,
              borderWidth: 1,
              borderColor: colors.border,
            }}
          >
            <Ionicons name="person" size={18} color={colors.accent} />
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}
