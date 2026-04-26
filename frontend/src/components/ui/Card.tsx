import { Pressable, View, ViewStyle } from "react-native";
import { colors } from "@/theme/colors";

type Props = {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  padded?: boolean;
  glow?: boolean;
};

export function Card({ children, onPress, style, padded = true, glow }: Props) {
  const base: ViewStyle = {
    backgroundColor: colors.bgCard,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    padding: padded ? 16 : 0,
    ...(glow
      ? {
          shadowColor: colors.accent,
          shadowOpacity: 0.15,
          shadowRadius: 14,
          shadowOffset: { width: 0, height: 4 },
        }
      : {}),
    ...style,
  };

  if (onPress) {
    return (
      <Pressable onPress={onPress} style={({ pressed }) => [base, pressed && { opacity: 0.85 }]}>
        {children}
      </Pressable>
    );
  }
  return <View style={base}>{children}</View>;
}
