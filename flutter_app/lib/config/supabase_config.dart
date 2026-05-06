/// Supabase project credentials. The anon key is *intended* to be
/// shipped client-side — RLS on the database is what actually protects
/// data. Rotate the key in the Supabase dashboard if it ever leaks
/// into the wrong hands.
class SupabaseConfig {
  SupabaseConfig._();

  static const String url = 'https://dcqbwnatncnylvbkhwav.supabase.co';
  static const String anonKey =
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRjcWJ3bmF0bmNueWx2Ymtod2F2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc3NDAxNzIsImV4cCI6MjA5MzMxNjE3Mn0.dXLzfbVwtOE8lDiuDtUTLsGd3So0cUH7MkmOLTc69ps';
}
