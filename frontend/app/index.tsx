import { Redirect } from "expo-router";

// Entry — the real redirect logic lives in _layout.tsx based on
// onboarding + auth state. This bounces anyone who lands here to
// the first sensible screen.
export default function Index() {
  return <Redirect href="/onboarding" />;
}
