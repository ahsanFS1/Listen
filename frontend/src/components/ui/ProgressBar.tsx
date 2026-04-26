import { LinearGradient } from "expo-linear-gradient";
import { View } from "react-native";
import { colors } from "@/theme/colors";

type Props = {
  value: number; // 0..1
  height?: number;
  gradient?: [string, string];
};

export function ProgressBar({
  value,
  height = 8,
  gradient = [colors.accent, colors.brandPurple],
}: Props) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <View
      style={{
        height,
        backgroundColor: colors.bgSoft,
        borderRadius: height / 2,
        overflow: "hidden",
      }}
    >
      <LinearGradient
        colors={gradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={{
          width: `${pct * 100}%`,
          height: "100%",
          borderRadius: height / 2,
        }}
      />
    </View>
  );
}
