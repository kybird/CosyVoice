/// Pipeline-wide logger using print() for guaranteed console visibility.
void pLog(String message, {String tag = 'Pipeline'}) {
  print('[$tag] $message');
}

/// Log a list of ints (truncated).
String shortList(List<int> list, {int max = 20}) {
  if (list.length <= max) return list.toString();
  return '[${list.sublist(0, max).join(", ")}, ...] (${list.length} total)';
}
