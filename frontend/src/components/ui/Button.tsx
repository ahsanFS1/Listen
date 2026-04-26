import { LinearGradient } from "expo-linear-gradient";
import { Pressable, Text, View, ActivityIndicator } from "react-native";
import { colors } from "@/theme/colors";

type Variant = "primary" | "ghost" | "outline";

type Props = {
  label: string;
  onPress?: () => void;
  variant?: Variant;
  icon?: React.ReactNode;
  trailingIcon?: React.ReactNode;
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
};

export function Button({
  label,
  onPress,
  variant = "primary",
  icon,
  trailingIcon,
  disabled,
  loading,
  fullWidth = true,
}: Props) {
  const content = (
    <View
      className="flex-row items-center justify-center gap-2"
      style={{
        paddingVertical: 16,
        paddingHorizontal: 24,
        borderRadius: 16,
      }}
    >
      {loading ? (
        <ActivityIndicator color={variant === "primary" ? "#0B1020" : colors.accent} />
      ) : (
        <>
          {icon}
          <Text
            style={{
              color: variant === "primary" ? "#0B1020" : colors.text,
              fontWeight: "700",
              fontSize: 16,
              letterSpacing: 0.3,
            }}
          >
            {label}
          </Text>
          {trailingIcon}
        </>
      )}
    </View>
  );

  if (variant === "primary") {
    return (
      <Pressable
        onPress={onPress}
        disabled={disabled || loading}
        style={{ opacity: disabled ? 0.5 : 1, width: fullWidth ? "100%" : undefined }}
      >
        <LinearGradient
          colors={[colors.accent, colors.accentSoft]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={{
            borderRadius: 16,
            shadowColor: colors.accent,
            shadowOpacity: 0.35,
            shadowRadius: 12,
            shadowOffset: { width: 0, height: 4 },
          }}
        >
          {content}
        </LinearGradient>
      </Pressable>
    );
  }

  const isOutline = variant === "outline";
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={{
        width: fullWidth ? "100%" : undefined,
        backgroundColor: isOutline ? "transparent" : colors.bgSoft,
        borderRadius: 16,
        borderWidth: isOutline ? 1 : 0,
        borderColor: colors.border,
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {content}
    </Pressable>
  );
}
