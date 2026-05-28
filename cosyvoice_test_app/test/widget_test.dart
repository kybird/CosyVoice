import 'package:flutter_test/flutter_test.dart';
import 'package:cosyvoice_test_app/main.dart';

void main() {
  testWidgets('App renders', (WidgetTester tester) async {
    await tester.pumpWidget(const CosyVoiceApp());
    expect(find.text('CosyVoice3 ONNX TTS'), findsOneWidget);
  });
}
