import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { CameraView, useCameraPermissions } from "expo-camera";
import { LinearGradient } from "expo-linear-gradient";
import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Animated,
  Easing,
  Pressable,
  ScrollView,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Header } from "@/components/Header";
import { useSignRecognition } from "@/hooks/useSignRecognition";
import { useTTS } from "@/hooks/useTTS";
import { useSessionStore } from "@/store/sessionStore";
import { colors } from "@/theme/colors";
import { PipelineMode } from "@/ml/pipeline";

type TranslateMode = "words" | "alphabets";

export default function TranslateScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraOn, setCameraOn] = useState(false);
  const [mode, setMode] = useState<TranslateMode>("words");
  const [showHistory, setShowHistory] = useState(false);
  const [ttsHint, setTtsHint] = useState<string | null>(null);
  const [lastCommittedWord, setLastCommittedWord] = useState<string | null>(null);
  const [isHoldingSign, setIsHoldingSign] = useState(false); // Hold-to-sign for demo mode
  // Some Android emulators report LENS_FACING=null on their virtual camera,
  // so requesting facing="front" returns no match and the preview renders
  // black. We start on "front" but flip to "back" automatically if the
  // preview never produces a valid surface.
  const [cameraFacing, setCameraFacing] = useState<"front" | "back">("front");
  const [cameraReady, setCameraReady] = useState(false);

  // Predictions flow when camera is on - both modes now work
  const pipelineMode: PipelineMode = mode === "words" ? "words" : "alphabet";
  const recognitionActive = cameraOn;
  const { prediction, reset, hasHands, isCommitted } = useSignRecognition(recognitionActive, pipelineMode, isHoldingSign);
  const { speak, supported: ttsSupported, voiceLanguage } = useTTS();
  const { history, commitWord, clearHistory, undoLast } = useSessionStore();
  
  // Track if alphabet model is available
  const [alphabetReady, setAlphabetReady] = useState(true); // Set to true since we added the model

  // Auto-commit on the pipeline's COMMITTED frame with auto-speak
  useEffect(() => {
    if (!prediction || !prediction.committed) return;
    if (!prediction.label) return;
    
    // Only commit if it's a new word (not the same as last committed)
    if (prediction.label !== lastCommittedWord) {
      commitWord({
        label: prediction.label,
        english: prediction.english,
        urdu: prediction.urdu,
      });
      setLastCommittedWord(prediction.label);
      
      // Auto-speak the committed word with priority
      if (ttsSupported) {
        speak(prediction.urdu, { priority: true });
      }
    }
  }, [prediction, commitWord, lastCommittedWord, speak, ttsSupported]);
  
  // Reset lastCommittedWord when camera stops
  useEffect(() => {
    if (!cameraOn) {
      setLastCommittedWord(null);
    }
  }, [cameraOn]);

  // If the front camera never reports ready within a short window, flip to back.
  // expo-camera silently renders a black surface when the requested facing
  // doesn't match any device camera (e.g. emulators where LENS_FACING is null).
  useEffect(() => {
    if (!cameraOn) {
      setCameraReady(false);
      setCameraFacing("front");
      return;
    }
    if (cameraReady || cameraFacing === "back") return;
    const t = setTimeout(() => {
      if (!cameraReady) {
        console.warn("[CameraView] front camera not ready, flipping to back");
        setCameraFacing("back");
      }
    }, 2500);
    return () => clearTimeout(t);
  }, [cameraOn, cameraReady, cameraFacing]);

  const onStart = async () => {
    if (!permission?.granted) {
      const r = await requestPermission();
      if (!r.granted) {
        Alert.alert(
          "Camera permission needed",
          "Listen needs camera access to recognize PSL signs.",
        );
        return;
      }
    }
    setCameraOn(true);
    setLastCommittedWord(null);
    reset();
  };

  const onStop = () => {
    setCameraOn(false);
    reset();
  };

  const onSpeak = (text: string) => {
    if (!ttsSupported) {
      setTtsHint("Speech requires the mobile app — install the dev build for TTS.");
      setTimeout(() => setTtsHint(null), 3500);
      return;
    }
    speak(text);
  };

  const state = prediction?.state ?? "IDLE";
  // Only show predictions when hands are actually visible
  const showingHands = hasHands && !!prediction;
  const pillColor =
    state === "PREDICTING" || state === "COMMITTED"
      ? colors.accent
      : state === "SIGNING"
      ? colors.warn
      : showingHands
      ? colors.ok
      : colors.textDim;
  const pillLabel = !cameraOn
    ? "OFFLINE"
    : !showingHands
    ? "SHOW HANDS"
    : state === "IDLE"
    ? "SCANNING"
    : state;

  return (
    <SafeAreaView edges={["top"]} style={{ flex: 1, backgroundColor: colors.bg }}>
      <Header />
      <ScrollView contentContainerStyle={{ paddingBottom: 30 }}>
        {/* Mode toggle */}
        <View style={{ paddingHorizontal: 18, marginTop: 4, marginBottom: 12 }}>
          <ModeToggle mode={mode} setMode={setMode} cameraOn={cameraOn} />
        </View>

        <View style={{ alignItems: "center" }}>
          <StatePill label={pillLabel} color={pillColor} animated={cameraOn && pillLabel !== "OFFLINE"} />
        </View>

        <View style={{ paddingHorizontal: 18, marginTop: 14 }}>
          <View
            style={{
              borderRadius: 24,
              overflow: "hidden",
              borderWidth: 1.5,
              borderColor: cameraOn ? colors.accent : colors.border,
              shadowColor: colors.accent,
              shadowOpacity: cameraOn ? 0.4 : 0,
              shadowRadius: 22,
              shadowOffset: { width: 0, height: 0 },
            }}
          >
            <View
              style={{
                aspectRatio: 3 / 4,
                backgroundColor: "#05070F",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              {cameraOn && permission?.granted ? (
                <CameraView
                  style={{ width: "100%", height: "100%" }}
                  facing={cameraFacing}
                  mirror={cameraFacing === "front"}
                  onCameraReady={() => setCameraReady(true)}
                  onMountError={(e) => {
                    console.warn("[CameraView] mount error", e);
                    if (cameraFacing === "front") {
                      setCameraFacing("back");
                      setCameraReady(false);
                    }
                  }}
                />
              ) : (
                <CameraIdle onStart={onStart} mode={mode} />
              )}

              {cameraOn ? <CornerBrackets /> : null}

              {cameraOn ? (
                <View
                  style={{
                    position: "absolute",
                    top: 14,
                    left: 14,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 6,
                    backgroundColor: "rgba(0,0,0,0.4)",
                    borderWidth: 1,
                    borderColor: `${colors.accent}60`,
                    borderRadius: 999,
                    paddingHorizontal: 10,
                    paddingVertical: 5,
                  }}
                >
                  <PulsingDot color={colors.accent} />
                  <Text style={{ color: colors.accent, fontWeight: "700", fontSize: 11, letterSpacing: 1 }}>
                    LIVE
                  </Text>
                </View>
              ) : null}

              {cameraOn ? (
                <>
                  {/* Hold to Sign button - for demo mode control */}
                  <Pressable
                    onPressIn={() => setIsHoldingSign(true)}
                    onPressOut={() => setIsHoldingSign(false)}
                    style={({ pressed }) => ({
                      position: "absolute",
                      top: 14,
                      left: 14,
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: pressed ? colors.accent : "rgba(0,0,0,0.55)",
                      borderWidth: 2,
                      borderColor: isHoldingSign ? colors.accent : colors.border,
                      paddingHorizontal: 12,
                      paddingVertical: 6,
                      borderRadius: 999,
                    })}
                  >
                    <Ionicons 
                      name={isHoldingSign ? "hand-left" : "hand-left-outline"} 
                      size={14} 
                      color={isHoldingSign ? "#000" : colors.text} 
                    />
                    <Text style={{ 
                      color: isHoldingSign ? "#000" : colors.text, 
                      fontSize: 12, 
                      fontWeight: "700" 
                    }}>
                      {isHoldingSign ? "SIGNING..." : "HOLD TO SIGN"}
                    </Text>
                  </Pressable>

                  <Pressable
                    onPress={onStop}
                    style={{
                      position: "absolute",
                      top: 14,
                      right: 14,
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: "rgba(0,0,0,0.55)",
                      borderWidth: 1,
                      borderColor: colors.border,
                      paddingHorizontal: 10,
                      paddingVertical: 5,
                      borderRadius: 999,
                    }}
                  >
                    <Ionicons name="stop" size={11} color={colors.text} />
                    <Text style={{ color: colors.text, fontSize: 11, fontWeight: "700" }}>
                      STOP
                    </Text>
                  </Pressable>
                </>
              ) : null}

              {/* "No hands" coaching when camera is on but we haven't seen a sign yet */}
              {cameraOn && !showingHands ? (
                <View
                  style={{
                    position: "absolute",
                    bottom: 18,
                    alignSelf: "center",
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 6,
                    backgroundColor: "rgba(0,0,0,0.55)",
                    borderRadius: 999,
                    paddingHorizontal: 14,
                    paddingVertical: 8,
                    borderWidth: 1,
                    borderColor: colors.border,
                  }}
                >
                  <MaterialCommunityIcons name="hand-wave" size={14} color={colors.warn} />
                  <Text style={{ color: colors.text, fontSize: 12 }}>
                    Show both hands to begin signing
                  </Text>
                </View>
              ) : null}
            </View>

            {/* Prediction readout */}
            <View
              style={{
                padding: 18,
                backgroundColor: colors.bgCard,
                borderTopWidth: 1,
                borderTopColor: colors.border,
              }}
            >
              <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                <View style={{ flex: 1 }}>
                  <Label>ENGLISH</Label>
                  <Text
                    style={{
                      color: showingHands ? colors.text : colors.textDim,
                      fontSize: 22,
                      fontWeight: "700",
                      marginTop: 4,
                    }}
                  >
                    {showingHands ? prediction!.english : "—"}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "flex-end" }}>
                  <Label>URDU</Label>
                  <Text
                    style={{
                      color: showingHands ? colors.text : colors.textDim,
                      fontSize: 24,
                      fontWeight: "700",
                      marginTop: 4,
                      writingDirection: "rtl",
                    }}
                  >
                    {showingHands ? prediction!.urdu : "—"}
                  </Text>
                </View>
              </View>
              <View style={{ marginTop: 14 }}>
                <View
                  style={{
                    flexDirection: "row",
                    justifyContent: "space-between",
                    marginBottom: 6,
                  }}
                >
                  <Label>CONFIDENCE</Label>
                  <Text
                    style={{
                      color: colors.accent,
                      fontWeight: "700",
                      fontSize: 13,
                    }}
                  >
                    {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "0%"}
                  </Text>
                </View>
                <ProgressBar value={prediction?.confidence ?? 0} height={6} />
              </View>
            </View>
          </View>
        </View>

        {/* Action row */}
        <View style={{ paddingHorizontal: 18, marginTop: 18 }}>
          <View style={{ flexDirection: "row", gap: 10 }}>
            <View style={{ flex: 1 }}>
              <Button
                label="Speak"
                icon={<Ionicons name="volume-high" size={18} color="#0B1020" />}
                disabled={!showingHands}
                onPress={() => prediction && onSpeak(prediction.urdu)}
              />
            </View>
            <IconButton
              onPress={clearHistory}
              icon={<Ionicons name="close" size={22} color={colors.text} />}
            />
            <IconButton
              onPress={() => setShowHistory((v) => !v)}
              icon={
                <Ionicons
                  name={showHistory ? "chevron-up" : "time"}
                  size={22}
                  color={colors.text}
                />
              }
            />
          </View>
          {ttsHint ? (
            <Text style={{ color: colors.warn, fontSize: 12, marginTop: 8 }}>
              {ttsHint}
            </Text>
          ) : null}
        </View>

        {/* Running sentence strip */}
        {history.length > 0 ? (
          <View style={{ paddingHorizontal: 18, marginTop: 18 }}>
            <Card>
              <Label>SENTENCE</Label>
              <Text
                style={{
                  color: colors.text,
                  fontSize: 22,
                  marginTop: 8,
                  writingDirection: "rtl",
                  textAlign: "right",
                  lineHeight: 32,
                }}
              >
                {history.map((h) => h.urdu).join(" ")}
              </Text>
              <Text style={{ color: colors.textDim, fontSize: 13, marginTop: 6 }}>
                {history.map((h) => h.english).join(" ")}
              </Text>
              <View style={{ flexDirection: "row", gap: 8, marginTop: 12 }}>
                <Pressable
                  onPress={undoLast}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 4,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                    borderRadius: 10,
                    backgroundColor: colors.bgSoft,
                    borderWidth: 1,
                    borderColor: colors.border,
                  }}
                >
                  <Ionicons name="arrow-undo" size={14} color={colors.text} />
                  <Text style={{ color: colors.text, fontSize: 12, fontWeight: "600" }}>
                    Undo
                  </Text>
                </Pressable>
                <Pressable
                  onPress={() => onSpeak(history.map((h) => h.urdu).join(" "))}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 4,
                    paddingHorizontal: 12,
                    paddingVertical: 8,
                    borderRadius: 10,
                    backgroundColor: `${colors.accent}20`,
                    borderWidth: 1,
                    borderColor: `${colors.accent}60`,
                  }}
                >
                  <Ionicons name="volume-high" size={14} color={colors.accent} />
                  <Text style={{ color: colors.accent, fontSize: 12, fontWeight: "700" }}>
                    Speak sentence
                  </Text>
                </Pressable>
              </View>
            </Card>
          </View>
        ) : null}

        {showHistory && history.length > 0 ? (
          <View style={{ paddingHorizontal: 18, marginTop: 14 }}>
            <Label>HISTORY</Label>
            <View style={{ gap: 8, marginTop: 8 }}>
              {history
                .slice()
                .reverse()
                .map((h) => (
                  <View
                    key={h.timestamp}
                    style={{
                      flexDirection: "row",
                      justifyContent: "space-between",
                      backgroundColor: colors.bgCard,
                      borderRadius: 12,
                      padding: 12,
                      borderWidth: 1,
                      borderColor: colors.border,
                    }}
                  >
                    <Text style={{ color: colors.text }}>{h.english}</Text>
                    <Text style={{ color: colors.accent, writingDirection: "rtl" }}>
                      {h.urdu}
                    </Text>
                  </View>
                ))}
            </View>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function ModeToggle({
  mode,
  setMode,
  cameraOn,
}: {
  mode: TranslateMode;
  setMode: (m: TranslateMode) => void;
  cameraOn: boolean;
}) {
  const Tab = ({ value, label, icon, badge }: { value: TranslateMode; label: string; icon: React.ReactNode; badge?: string }) => {
    const active = mode === value;
    return (
      <Pressable
        onPress={() => {
          if (cameraOn) return; // can't change mode while running
          setMode(value);
        }}
        style={{
          flex: 1,
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          paddingVertical: 11,
          borderRadius: 12,
          backgroundColor: active ? colors.bgSoft : "transparent",
          borderWidth: active ? 1 : 0,
          borderColor: active ? colors.accent : "transparent",
          opacity: cameraOn && !active ? 0.4 : 1,
        }}
      >
        {icon}
        <Text
          style={{
            color: active ? colors.accent : colors.textDim,
            fontWeight: "700",
            fontSize: 13,
            letterSpacing: 1,
          }}
        >
          {label}
        </Text>
        {badge ? (
          <View
            style={{
              backgroundColor: colors.warn,
              paddingHorizontal: 6,
              paddingVertical: 2,
              borderRadius: 6,
              marginLeft: 2,
            }}
          >
            <Text style={{ color: "#0B1020", fontSize: 9, fontWeight: "800" }}>
              {badge}
            </Text>
          </View>
        ) : null}
      </Pressable>
    );
  };
  return (
    <View
      style={{
        flexDirection: "row",
        backgroundColor: colors.bgCard,
        borderRadius: 14,
        padding: 4,
        borderWidth: 1,
        borderColor: colors.border,
      }}
    >
      <Tab
        value="words"
        label="WORDS"
        icon={<MaterialCommunityIcons name="text-box" size={16} color={mode === "words" ? colors.accent : colors.textDim} />}
      />
      <Tab
        value="alphabets"
        label="ALPHABETS"
        icon={<MaterialCommunityIcons name="alphabetical-variant" size={16} color={mode === "alphabets" ? colors.accent : colors.textDim} />}
      />
    </View>
  );
}

function CameraIdle({ onStart, mode }: { onStart: () => void; mode: TranslateMode }) {
  return (
    <LinearGradient
      colors={["#101A40", "#05070F"]}
      style={{
        flex: 1,
        width: "100%",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        gap: 18,
      }}
    >
      <View
        style={{
          width: 84,
          height: 84,
          borderRadius: 42,
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: colors.bgSoft,
          borderWidth: 2,
          borderColor: colors.accent,
        }}
      >
        <Ionicons name="videocam" size={36} color={colors.accent} />
      </View>
      <Text style={{ color: colors.text, fontSize: 18, fontWeight: "700", textAlign: "center" }}>
        Tap start to begin live translation
      </Text>
      <Text style={{ color: colors.textDim, textAlign: "center", lineHeight: 20 }}>
        {mode === "words"
          ? "Sign one word at a time, holding each gesture for ~1 second."
          : "Sign each Urdu alphabet letter clearly. Build words letter by letter!"}
      </Text>
      <Pressable
        onPress={onStart}
        style={{
          flexDirection: "row",
          alignItems: "center",
          gap: 8,
          paddingHorizontal: 22,
          paddingVertical: 12,
          borderRadius: 999,
          backgroundColor: colors.accent,
        }}
      >
        <Ionicons name="play" size={16} color="#0B1020" />
        <Text style={{ color: "#0B1020", fontWeight: "800", letterSpacing: 0.5 }}>
          Start camera
        </Text>
      </Pressable>
    </LinearGradient>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <Text
      style={{
        color: colors.textDim,
        fontSize: 10,
        fontWeight: "700",
        letterSpacing: 1.6,
      }}
    >
      {children}
    </Text>
  );
}

function StatePill({
  label,
  color,
  animated,
}: {
  label: string;
  color: string;
  animated: boolean;
}) {
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        backgroundColor: colors.bgSoft,
        borderWidth: 1,
        borderColor: color,
        paddingHorizontal: 14,
        paddingVertical: 6,
        borderRadius: 999,
      }}
    >
      {animated ? (
        <PulsingDot color={color} />
      ) : (
        <View style={{ width: 7, height: 7, borderRadius: 4, backgroundColor: color, opacity: 0.8 }} />
      )}
      <Text style={{ color, fontWeight: "700", fontSize: 12, letterSpacing: 1.5 }}>
        {label}
      </Text>
    </View>
  );
}

function PulsingDot({ color }: { color: string }) {
  const anim = useMemo(() => new Animated.Value(0), []);
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(anim, {
          toValue: 1,
          duration: 700,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(anim, {
          toValue: 0,
          duration: 700,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [anim]);
  return (
    <Animated.View
      style={{
        width: 7,
        height: 7,
        borderRadius: 4,
        backgroundColor: color,
        opacity: anim.interpolate({ inputRange: [0, 1], outputRange: [0.4, 1] }),
        transform: [
          {
            scale: anim.interpolate({ inputRange: [0, 1], outputRange: [0.9, 1.15] }),
          },
        ],
      }}
    />
  );
}

function IconButton({
  icon,
  onPress,
}: {
  icon: React.ReactNode;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={{
        width: 52,
        height: 52,
        borderRadius: 16,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: colors.bgSoft,
        borderWidth: 1,
        borderColor: colors.border,
      }}
    >
      {icon}
    </Pressable>
  );
}

function CornerBrackets() {
  const bracket = {
    position: "absolute" as const,
    width: 22,
    height: 22,
    borderColor: colors.accent,
  };
  return (
    <>
      <View style={{ ...bracket, top: 10, left: 10, borderTopWidth: 2, borderLeftWidth: 2 }} />
      <View style={{ ...bracket, top: 10, right: 10, borderTopWidth: 2, borderRightWidth: 2 }} />
      <View style={{ ...bracket, bottom: 10, left: 10, borderBottomWidth: 2, borderLeftWidth: 2 }} />
      <View style={{ ...bracket, bottom: 10, right: 10, borderBottomWidth: 2, borderRightWidth: 2 }} />
    </>
  );
}
