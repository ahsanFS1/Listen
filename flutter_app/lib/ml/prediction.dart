enum SignState { idle, signing, predicting, committed, cooldown }

class Prediction {
  final String label;
  final String english;
  final String urdu;
  final double confidence;
  final SignState state;
  final bool committed;
  final bool hasHands;

  const Prediction({
    required this.label,
    required this.english,
    required this.urdu,
    required this.confidence,
    required this.state,
    required this.committed,
    required this.hasHands,
  });

  static const Prediction idle = Prediction(
    label: '', english: '\u2014', urdu: '\u2014',
    confidence: 0, state: SignState.idle,
    committed: false, hasHands: false,
  );
}
