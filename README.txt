PTT combo binding fix:
- Change can capture modifier+key combos such as Alt+Z
- Modifier alone waits for the next key
- Runtime PTT checks modifier and key together
- Config stores ptt_mod and ptt_vk
- Existing UI/history/unicode/icon/live-language/Small INT8 model/tail=-1 retained


Required model files:
- models/small-encoder.int8.onnx
- models/small-decoder.int8.onnx
- models/small-tokens.txt


PZ text bridge
--------------
SpeechHelper publishes the latest recognized text to:
  %USERPROFILE%\Zomboid\Lua\SpeechHelper.txt
The file is replaced atomically after each successful transcription. The Project Zomboid mod can poll this file with getFileReader and consume new lines.

