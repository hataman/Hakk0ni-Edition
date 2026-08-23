
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <mmsystem.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"
#include <fstream>
#include <cstdio>

#pragma comment(lib, "winmm.lib")

namespace fs = std::filesystem;

static constexpr int kSampleRate = 16000;
static constexpr int kChannels = 1;
static constexpr int kBitsPerSample = 16;
static constexpr int kBufferMs = 100;
static constexpr int kSamplesPerBuffer = kSampleRate * kBufferMs / 1000;
static constexpr int kBufferCount = 8;


struct WaveBuffer {
    WAVEHDR hdr{};
    std::vector<int16_t> data;
};

class AudioCapture {
public:
    AudioCapture() : buffers_(kBufferCount) {}

    ~AudioCapture() {
        if (active_) {
            Stop();
        }
    }

    bool Start() {
        if (active_) return false;

        WAVEFORMATEX fmt{};
        fmt.wFormatTag = WAVE_FORMAT_PCM;
        fmt.nChannels = kChannels;
        fmt.nSamplesPerSec = kSampleRate;
        fmt.wBitsPerSample = kBitsPerSample;
        fmt.nBlockAlign = fmt.nChannels * fmt.wBitsPerSample / 8;
        fmt.nAvgBytesPerSec = fmt.nSamplesPerSec * fmt.nBlockAlign;

        MMRESULT r = waveInOpen(
            &waveIn_,
            WAVE_MAPPER,
            &fmt,
            reinterpret_cast<DWORD_PTR>(&AudioCapture::WaveInProc),
            reinterpret_cast<DWORD_PTR>(this),
            CALLBACK_FUNCTION
        );

        if (r != MMSYSERR_NOERROR) {
            std::cerr << "[MIC] waveInOpen failed: " << r << "\n";
            waveIn_ = nullptr;
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            samples_.clear();
        }

        active_.store(true);

        for (auto& b : buffers_) {
            b.data.assign(kSamplesPerBuffer, 0);
            b.hdr = {};
            b.hdr.lpData = reinterpret_cast<LPSTR>(b.data.data());
            b.hdr.dwBufferLength =
                static_cast<DWORD>(b.data.size() * sizeof(int16_t));

            waveInPrepareHeader(waveIn_, &b.hdr, sizeof(WAVEHDR));
            waveInAddBuffer(waveIn_, &b.hdr, sizeof(WAVEHDR));
        }

        r = waveInStart(waveIn_);
        if (r != MMSYSERR_NOERROR) {
            std::cerr << "[MIC] waveInStart failed: " << r << "\n";
            Stop();
            return false;
        }

        return true;
    }

    std::vector<float> Stop() {
        if (!waveIn_) return {};

        active_.store(false);

        waveInStop(waveIn_);
        waveInReset(waveIn_);

        for (auto& b : buffers_) {
            waveInUnprepareHeader(waveIn_, &b.hdr, sizeof(WAVEHDR));
        }

        waveInClose(waveIn_);
        waveIn_ = nullptr;

        std::vector<int16_t> pcm16;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pcm16 = samples_;
            samples_.clear();
        }

        std::vector<float> out;
        out.reserve(pcm16.size());

        constexpr float scale = 1.0f / 32768.0f;
        for (int16_t s : pcm16) {
            out.push_back(static_cast<float>(s) * scale);
        }

        return out;
    }

private:
    static void CALLBACK WaveInProc(
        HWAVEIN hwi,
        UINT msg,
        DWORD_PTR instance,
        DWORD_PTR param1,
        DWORD_PTR
    ) {
        if (msg != WIM_DATA) return;

        auto* self = reinterpret_cast<AudioCapture*>(instance);
        auto* hdr = reinterpret_cast<WAVEHDR*>(param1);

        if (!self || !hdr) return;

        if (self->active_.load() && hdr->dwBytesRecorded > 0) {
            const auto* data =
                reinterpret_cast<const int16_t*>(hdr->lpData);

            const size_t count =
                hdr->dwBytesRecorded / sizeof(int16_t);

            {
                std::lock_guard<std::mutex> lock(self->mutex_);
                self->samples_.insert(
                    self->samples_.end(),
                    data,
                    data + count
                );
            }
        }

        if (self->active_.load()) {
            hdr->dwBytesRecorded = 0;
            waveInAddBuffer(hwi, hdr, sizeof(WAVEHDR));
        }
    }

    HWAVEIN waveIn_ = nullptr;
    std::atomic<bool> active_{false};
    std::mutex mutex_;
    std::vector<int16_t> samples_;
    std::vector<WaveBuffer> buffers_;
};

static std::string LoadLanguageCode() {
    std::ifstream f("config.txt");
    if (!f) return "tr";

    std::string line;
    while (std::getline(f, line)) {
        const std::string key = "language=";
        if (line.rfind(key, 0) == 0) {
            std::string value = line.substr(key.size());
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            const auto end = value.find_last_not_of(" \t\r\n");
            if (end != std::string::npos) value.erase(end + 1);
            if (!value.empty()) return value;
        }
    }
    return "tr";
}

class SpeechRecognizer {
public:
    ~SpeechRecognizer() {
        Shutdown();
    }

    bool Init(
        const fs::path& encoder,
        const fs::path& decoder,
        const fs::path& tokens
    ) {
        Shutdown();

        encoder_ = encoder.string();
        decoder_ = decoder.string();
        tokens_ = tokens.string();

        SherpaOnnxOfflineRecognizerConfig config{};
        config.feat_config.sample_rate = kSampleRate;
        config.feat_config.feature_dim = 80;

        config.decoding_method = "greedy_search";

        config.model_config.debug = 0;
        config.model_config.num_threads = 1;
        config.model_config.provider = "cpu";
        config.model_config.tokens = tokens_.c_str();

        config.model_config.whisper.encoder = encoder_.c_str();
        config.model_config.whisper.decoder = decoder_.c_str();

        // Empty language = no forced language hint.
        // We want multilingual behavior in the first prototype.
        const std::string language = LoadLanguageCode();
        config.model_config.whisper.language = language.c_str();
        std::cout << "Language: " << language << "\n";
        config.model_config.whisper.task = "transcribe";
        config.model_config.whisper.tail_paddings = -1;

        recognizer_ = SherpaOnnxCreateOfflineRecognizer(&config);

        if (!recognizer_) {
            std::cerr << "[STT] recognizer init failed\n";
            return false;
        }

        return true;
    }


    bool ReloadLanguage(const std::string& language) {
        Shutdown();

        SherpaOnnxOfflineRecognizerConfig config{};
        config.feat_config.sample_rate = kSampleRate;
        config.feat_config.feature_dim = 80;

        config.decoding_method = "greedy_search";

        config.model_config.debug = 0;
        config.model_config.num_threads = 1;
        config.model_config.provider = "cpu";
        config.model_config.tokens = tokens_.c_str();

        config.model_config.whisper.encoder = encoder_.c_str();
        config.model_config.whisper.decoder = decoder_.c_str();
        config.model_config.whisper.language = language.c_str();
        config.model_config.whisper.task = "transcribe";
        config.model_config.whisper.tail_paddings = -1;

        recognizer_ = SherpaOnnxCreateOfflineRecognizer(&config);

        if (!recognizer_) {
            return false;
        }

        return true;
    }

    std::string Transcribe(const std::vector<float>& samples) {
        if (!recognizer_ || samples.empty()) return {};

        const SherpaOnnxOfflineStream* stream =
            SherpaOnnxCreateOfflineStream(recognizer_);

        if (!stream) {
            std::cerr << "[STT] stream creation failed\n";
            return {};
        }

        SherpaOnnxAcceptWaveformOffline(
            stream,
            kSampleRate,
            samples.data(),
            static_cast<int32_t>(samples.size())
        );

        SherpaOnnxDecodeOfflineStream(recognizer_, stream);

        const SherpaOnnxOfflineRecognizerResult* result =
            SherpaOnnxGetOfflineStreamResult(stream);

        std::string text;

        if (result && result->text) {
            text = result->text;
        }

        if (result) {
            SherpaOnnxDestroyOfflineRecognizerResult(result);
        }

        SherpaOnnxDestroyOfflineStream(stream);

        return text;
    }

private:
    void Shutdown() {
        if (recognizer_) {
            SherpaOnnxDestroyOfflineRecognizer(recognizer_);
            recognizer_ = nullptr;
        }
    }

    const SherpaOnnxOfflineRecognizer* recognizer_ = nullptr;
    std::string encoder_;
    std::string decoder_;
    std::string tokens_;
};

static fs::path ExeDir() {
    char buffer[MAX_PATH]{};
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    return fs::path(buffer).parent_path();
}





static constexpr UINT WM_APP_STT_TEXT = WM_APP + 1;
static constexpr UINT WM_APP_STT_STATUS = WM_APP + 2;
static HWND g_mainWindow = nullptr;
static HWND g_history = nullptr;
static HWND g_status = nullptr;

static std::mutex g_languageMutex;
static std::string g_requestedLanguage = "tr";
static std::atomic<bool> g_languageReloadRequested{false};

enum HotkeyMods : int {
    HOTKEY_MOD_NONE  = 0,
    HOTKEY_MOD_CTRL  = 1 << 0,
    HOTKEY_MOD_SHIFT = 1 << 1,
    HOTKEY_MOD_ALT   = 1 << 2,
    HOTKEY_MOD_WIN   = 1 << 3,
};

static std::atomic<int> g_hotkeyMods{HOTKEY_MOD_NONE};
static std::atomic<int> g_hotkeyVk{VK_LMENU};

static std::atomic<bool> g_capturingHotkey{false};
static HWND g_hotkeyValue = nullptr;

static constexpr UINT_PTR HOTKEY_CAPTURE_TIMER = 1;


static bool IsModifierVk(int vk) {
    return vk == VK_SHIFT || vk == VK_LSHIFT || vk == VK_RSHIFT ||
           vk == VK_CONTROL || vk == VK_LCONTROL || vk == VK_RCONTROL ||
           vk == VK_MENU || vk == VK_LMENU || vk == VK_RMENU ||
           vk == VK_LWIN || vk == VK_RWIN;
}

static int CurrentModifierMask() {
    int mods = HOTKEY_MOD_NONE;

    if ((GetAsyncKeyState(VK_LCONTROL) & 0x8000) ||
        (GetAsyncKeyState(VK_RCONTROL) & 0x8000)) {
        mods |= HOTKEY_MOD_CTRL;
    }

    if ((GetAsyncKeyState(VK_LSHIFT) & 0x8000) ||
        (GetAsyncKeyState(VK_RSHIFT) & 0x8000)) {
        mods |= HOTKEY_MOD_SHIFT;
    }

    if ((GetAsyncKeyState(VK_LMENU) & 0x8000) ||
        (GetAsyncKeyState(VK_RMENU) & 0x8000)) {
        mods |= HOTKEY_MOD_ALT;
    }

    if ((GetAsyncKeyState(VK_LWIN) & 0x8000) ||
        (GetAsyncKeyState(VK_RWIN) & 0x8000)) {
        mods |= HOTKEY_MOD_WIN;
    }

    return mods;
}

static bool AnyCaptureKeyDown() {
    for (int vk = 1; vk < 256; ++vk) {
        // Left/right mouse are only UI clicks here; don't capture them.
        if (vk == VK_LBUTTON || vk == VK_RBUTTON) continue;

        if (GetAsyncKeyState(vk) & 0x8000) {
            return true;
        }
    }

    return false;
}

static int FindCaptureMainKey() {
    for (int vk = 1; vk < 256; ++vk) {
        if (vk == VK_LBUTTON || vk == VK_RBUTTON) continue;

        // Critical rule: modifier keys can never become the main key.
        if (IsModifierVk(vk)) continue;

        if (GetAsyncKeyState(vk) & 0x8000) {
            return vk;
        }
    }

    return 0;
}

static bool ModifierMaskDown(int mods) {
    if ((mods & HOTKEY_MOD_CTRL) &&
        !((GetAsyncKeyState(VK_LCONTROL) & 0x8000) ||
          (GetAsyncKeyState(VK_RCONTROL) & 0x8000))) {
        return false;
    }

    if ((mods & HOTKEY_MOD_SHIFT) &&
        !((GetAsyncKeyState(VK_LSHIFT) & 0x8000) ||
          (GetAsyncKeyState(VK_RSHIFT) & 0x8000))) {
        return false;
    }

    if ((mods & HOTKEY_MOD_ALT) &&
        !((GetAsyncKeyState(VK_LMENU) & 0x8000) ||
          (GetAsyncKeyState(VK_RMENU) & 0x8000))) {
        return false;
    }

    if ((mods & HOTKEY_MOD_WIN) &&
        !((GetAsyncKeyState(VK_LWIN) & 0x8000) ||
          (GetAsyncKeyState(VK_RWIN) & 0x8000))) {
        return false;
    }

    return true;
}
static std::string HotkeyKeyName(int vk) {
    switch (vk) {
    case VK_MENU: return "ALT";
    case VK_LMENU: return "Left ALT";
    case VK_RMENU: return "Right ALT";
    case VK_CONTROL: return "CTRL";
    case VK_LCONTROL: return "Left CTRL";
    case VK_RCONTROL: return "Right CTRL";
    case VK_SHIFT: return "SHIFT";
    case VK_LSHIFT: return "Left SHIFT";
    case VK_RSHIFT: return "Right SHIFT";
    case VK_LWIN: return "Left WIN";
    case VK_RWIN: return "Right WIN";
    case VK_CAPITAL: return "Caps Lock";
    case VK_SPACE: return "Space";
    case VK_TAB: return "Tab";
    case VK_MBUTTON: return "Mouse Middle";
    case VK_XBUTTON1: return "Mouse 4";
    case VK_XBUTTON2: return "Mouse 5";
    default: break;
    }

    char name[64]{};
    UINT scan = MapVirtualKeyA(static_cast<UINT>(vk), MAPVK_VK_TO_VSC) << 16;

    if (GetKeyNameTextA(static_cast<LONG>(scan), name, sizeof(name)) > 0) {
        return name;
    }

    char fallback[64]{};
    snprintf(fallback, sizeof(fallback), "VK %d", vk);
    return fallback;
}

static std::string HotkeyBindingName(int mods, int key) {
    std::string out;

    auto add = [&out](const char* value) {
        if (!out.empty()) out += " + ";
        out += value;
    };

    if (mods & HOTKEY_MOD_CTRL) add("CTRL");
    if (mods & HOTKEY_MOD_SHIFT) add("SHIFT");
    if (mods & HOTKEY_MOD_ALT) add("ALT");
    if (mods & HOTKEY_MOD_WIN) add("WIN");

    if (key != 0) {
        if (!out.empty()) out += " + ";
        out += HotkeyKeyName(key);
    }

    return out.empty() ? "None" : out;
}

static bool IsSpeechHotkeyDown() {
    const int mods = g_hotkeyMods.load();
    const int key = g_hotkeyVk.load();

    if (key == 0) return false;
    if (!ModifierMaskDown(mods)) return false;
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}

static int LegacyModifierVkToMask(int vk) {
    switch (vk) {
    case VK_CONTROL:
    case VK_LCONTROL:
    case VK_RCONTROL: return HOTKEY_MOD_CTRL;
    case VK_SHIFT:
    case VK_LSHIFT:
    case VK_RSHIFT: return HOTKEY_MOD_SHIFT;
    case VK_MENU:
    case VK_LMENU:
    case VK_RMENU: return HOTKEY_MOD_ALT;
    case VK_LWIN:
    case VK_RWIN: return HOTKEY_MOD_WIN;
    default: return HOTKEY_MOD_NONE;
    }
}

static int LoadHotkeyVk() {
    std::ifstream f("config.txt");
    if (!f) return VK_LMENU;

    int legacyVk = VK_LMENU;
    std::string line;

    while (std::getline(f, line)) {
        const std::string newKey = "hotkey_vk=";
        const std::string oldKey = "ptt_vk=";

        try {
            if (line.rfind(newKey, 0) == 0) {
                int value = std::stoi(line.substr(newKey.size()));
                if (value > 0 && value < 256) return value;
            }

            if (line.rfind(oldKey, 0) == 0) {
                int value = std::stoi(line.substr(oldKey.size()));
                if (value > 0 && value < 256) legacyVk = value;
            }
        } catch (...) {}
    }

    return legacyVk;
}

static int LoadHotkeyMods() {
    std::ifstream f("config.txt");
    if (!f) return HOTKEY_MOD_NONE;

    int legacyModVk = 0;
    std::string line;

    while (std::getline(f, line)) {
        const std::string newKey = "hotkey_mods=";
        const std::string oldKey = "ptt_mod=";

        try {
            if (line.rfind(newKey, 0) == 0) {
                int value = std::stoi(line.substr(newKey.size()));
                if (value >= 0 && value <= 15) return value;
            }

            if (line.rfind(oldKey, 0) == 0) {
                int value = std::stoi(line.substr(oldKey.size()));
                if (value >= 0 && value < 256) legacyModVk = value;
            }
        } catch (...) {}
    }

    return LegacyModifierVkToMask(legacyModVk);
}

static void WriteConfig(const std::string& language) {
    std::ofstream f("config.txt", std::ios::trunc);
    f << "language=" << language << "\n";
    f << "hotkey_mods=" << g_hotkeyMods.load() << "\n";
    f << "hotkey_vk=" << g_hotkeyVk.load() << "\n";
}

static void RequestLanguageReload(const std::string& code) {
    {
        std::lock_guard<std::mutex> lock(g_languageMutex);
        g_requestedLanguage = code;
    }
    g_languageReloadRequested.store(true);
}


static void PostUiStatus(const char* value) {
    if (!g_mainWindow) return;
    char* copy = _strdup(value ? value : "");
    PostMessageA(g_mainWindow, WM_APP_STT_STATUS, 0, reinterpret_cast<LPARAM>(copy));
}

static std::wstring Utf8ToWide(const std::string& value) {
    if (value.empty()) return L"";

    const int needed = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS,
        value.c_str(), static_cast<int>(value.size()),
        nullptr, 0
    );

    if (needed <= 0) return L"";

    std::wstring out(static_cast<size_t>(needed), L'\0');

    MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS,
        value.c_str(), static_cast<int>(value.size()),
        out.data(), needed
    );

    return out;
}

static void PostUiText(const std::string& value) {
    if (!g_mainWindow || value.empty()) return;

    std::wstring wide = Utf8ToWide(value);
    if (wide.empty()) return;

    const size_t bytes = (wide.size() + 1) * sizeof(wchar_t);
    auto* copy = static_cast<wchar_t*>(malloc(bytes));
    if (!copy) return;

    memcpy(copy, wide.c_str(), bytes);
    PostMessageA(g_mainWindow, WM_APP_STT_TEXT, 0, reinterpret_cast<LPARAM>(copy));
}

static bool SendTextToFocusedInput(const std::string& value) {
    const std::wstring wide = Utf8ToWide(value);
    if (wide.empty()) return false;

    std::vector<INPUT> inputs;
    inputs.reserve(wide.size() * 2);

    for (wchar_t ch : wide) {
        INPUT down{};
        down.type = INPUT_KEYBOARD;
        down.ki.wScan = ch;
        down.ki.dwFlags = KEYEVENTF_UNICODE;
        inputs.push_back(down);

        INPUT up = down;
        up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
        inputs.push_back(up);
    }

    const UINT sent = SendInput(
        static_cast<UINT>(inputs.size()),
        inputs.data(),
        sizeof(INPUT)
    );

    return sent == inputs.size();
}

static bool IsNonSpeechLabel(const std::string& raw) {
    std::string s = raw;
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\t'), s.end());

    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();

    if (s.empty()) return true;

    // Whisper can emit incomplete captions such as "[Music playing"
    // without a closing bracket. Any caption-like result is not speech.
    if (s.front() == '[' || s.front() == '(') return true;

    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    static const char* labels[] = {
        "music",
        "applause",
        "laughter",
        "silence",
        "noise"
    };

    for (const char* label : labels) {
        if (lower == label) return true;
    }

    return false;
}

enum : int {
    IDC_LANGUAGE = 1001,
    IDC_SAVE = 1002,
    IDC_STATUS = 1003,
    IDC_HISTORY = 1004,
    IDC_CLEAR = 1005,

    IDC_NORMAL_CHANGE = 1010
};

static const struct {
    const char* label;
    const char* code;
} kLanguages[] = {
    {"Turkish", "tr"},
    {"English", "en"},
    {"German", "de"},
    {"French", "fr"},
    {"Spanish", "es"},
    {"Italian", "it"},
    {"Portuguese", "pt"},
    {"Russian", "ru"},
    {"Polish", "pl"},
    {"Japanese", "ja"},
    {"Korean", "ko"},
    {"Chinese", "zh"},
};

static void SaveLanguageCode(const std::string& code) {
    WriteConfig(code);
    RequestLanguageReload(code);
}

static void RefreshHotkeyLabelFromSavedBinding() {
    if (!g_hotkeyValue) return;

    const std::string label =
        HotkeyBindingName(g_hotkeyMods.load(), g_hotkeyVk.load());
    SetWindowTextA(g_hotkeyValue, label.c_str());
}

static void SaveHotkeyBinding(int mods, int vk) {
    g_hotkeyMods.store(mods);
    g_hotkeyVk.store(vk);

    WriteConfig(LoadLanguageCode());
    RefreshHotkeyLabelFromSavedBinding();
}

static int FindLanguageIndex(const std::string& code) {
    for (int i = 0; i < static_cast<int>(sizeof(kLanguages) / sizeof(kLanguages[0])); ++i) {
        if (code == kLanguages[i].code) return i;
    }
    return 0;
}

static LRESULT CALLBACK SettingsWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    static HWND combo = nullptr;
    static HWND status = nullptr;
    static HWND history = nullptr;
    static HWND clearButton = nullptr;
    static bool captureWaitingForRelease = false;

    switch (msg) {
    case WM_APP_STT_TEXT: {
        wchar_t* value = reinterpret_cast<wchar_t*>(lParam);
        if (g_history && value) {
            const int len = GetWindowTextLengthW(g_history);
            SendMessageW(g_history, EM_SETSEL, len, len);

            if (len > 0) {
                SendMessageW(
                    g_history,
                    EM_REPLACESEL,
                    FALSE,
                    reinterpret_cast<LPARAM>(L"\r\n")
                );
            }

            SendMessageW(
                g_history,
                EM_REPLACESEL,
                FALSE,
                reinterpret_cast<LPARAM>(value)
            );

            SendMessageW(g_history, EM_SCROLLCARET, 0, 0);
        }
        free(value);
        return 0;
    }
    case WM_APP_STT_STATUS: {
        char* value = reinterpret_cast<char*>(lParam);
        if (g_status && value) SetWindowTextA(g_status, value);
        free(value);
        return 0;
    }

    case WM_CREATE: {
        CreateWindowA("STATIC", "Language:",
            WS_CHILD | WS_VISIBLE,
            20, 18, 80, 20,
            hwnd, nullptr, nullptr, nullptr);

        combo = CreateWindowA("COMBOBOX", "",
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
            100, 14, 200, 250,
            hwnd, reinterpret_cast<HMENU>(IDC_LANGUAGE), nullptr, nullptr);

        for (const auto& lang : kLanguages) {
            SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(lang.label));
        }

        const std::string current = LoadLanguageCode();
        SendMessageA(combo, CB_SETCURSEL, FindLanguageIndex(current), 0);

        CreateWindowA("BUTTON", "Apply",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            315, 14, 80, 25,
            hwnd, reinterpret_cast<HMENU>(IDC_SAVE), nullptr, nullptr);

        CreateWindowA("STATIC", "Talk:",
            WS_CHILD | WS_VISIBLE,
            20, 54, 45, 20,
            hwnd, nullptr, nullptr, nullptr);

        g_hotkeyValue = CreateWindowA(
            "STATIC",
            HotkeyBindingName(g_hotkeyMods.load(), g_hotkeyVk.load()).c_str(),
            WS_CHILD | WS_VISIBLE,
            70, 54, 120, 20,
            hwnd, nullptr, nullptr, nullptr);

        CreateWindowA("BUTTON", "Change",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            195, 50, 80, 25,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_NORMAL_CHANGE)),
            nullptr, nullptr);

        CreateWindowA("STATIC", "Status:",
            WS_CHILD | WS_VISIBLE,
            20, 94, 80, 20,
            hwnd, nullptr, nullptr, nullptr);

        status = CreateWindowA("STATIC", "Ready",
            WS_CHILD | WS_VISIBLE,
            100, 94, 210, 20,
            hwnd, reinterpret_cast<HMENU>(IDC_STATUS), nullptr, nullptr);
        g_status = status;

        CreateWindowA("STATIC", "Conversation history:",
            WS_CHILD | WS_VISIBLE,
            20, 128, 180, 20,
            hwnd, nullptr, nullptr, nullptr);

        RECT rc{};
        GetClientRect(hwnd, &rc);

        history = CreateWindowExW(
            WS_EX_CLIENTEDGE, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_VSCROLL |
            ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | ES_WANTRETURN,
            20, 152,
            (std::max)(200L, rc.right - 40),
            (std::max)(120L, rc.bottom - 202),
            hwnd, reinterpret_cast<HMENU>(IDC_HISTORY), nullptr, nullptr);
        g_history = history;

        clearButton = CreateWindowA("BUTTON", "Clear history",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20, (std::max)(156L, rc.bottom - 40), 110, 30,
            hwnd, reinterpret_cast<HMENU>(IDC_CLEAR), nullptr, nullptr);

        return 0;
    }

    case WM_SIZE: {
        const int clientW = LOWORD(lParam);
        const int clientH = HIWORD(lParam);

        if (history) {
            MoveWindow(
                history,
                20, 152,
                (std::max)(200, clientW - 40),
                (std::max)(120, clientH - 202),
                TRUE
            );
        }

        if (clearButton) {
            MoveWindow(
                clearButton,
                20, (std::max)(156, clientH - 40),
                110, 30,
                TRUE
            );
        }

        return 0;
    }

    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        if (info) {
            info->ptMinTrackSize.x = 580;
            info->ptMinTrackSize.y = 420;
        }
        return 0;
    }

    case WM_TIMER:
        if (wParam == HOTKEY_CAPTURE_TIMER && g_capturingHotkey.load()) {
            if (captureWaitingForRelease) {
                if (!AnyCaptureKeyDown()) {
                    captureWaitingForRelease = false;
                    if (g_hotkeyValue)
                        SetWindowTextA(g_hotkeyValue, "Press shortcut...");
                }
                return 0;
            }

            const int mods = CurrentModifierMask();
            const int key = FindCaptureMainKey();

            if (key == 0) {
                if (mods != HOTKEY_MOD_NONE && g_hotkeyValue) {
                    const std::string preview =
                        HotkeyBindingName(mods, 0) + " + ...";
                    SetWindowTextA(g_hotkeyValue, preview.c_str());
                }
                return 0;
            }

            SaveHotkeyBinding(mods, key);

            g_capturingHotkey.store(false);
            captureWaitingForRelease = false;
            KillTimer(hwnd, HOTKEY_CAPTURE_TIMER);
            return 0;
        }
        break;

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYUP:
    case WM_MBUTTONDOWN:
    case WM_XBUTTONDOWN:
        if (g_capturingHotkey.load()) {
            // Capture is handled only by WM_TIMER + GetAsyncKeyState.
            return 0;
        }
        break;

    case WM_COMMAND: {
        const int commandId = LOWORD(wParam);

        if (commandId == IDC_NORMAL_CHANGE) {
            captureWaitingForRelease = true;
            g_capturingHotkey.store(true);

            if (g_hotkeyValue)
                SetWindowTextA(g_hotkeyValue, "Release, then press...");

            SetTimer(hwnd, HOTKEY_CAPTURE_TIMER, 15, nullptr);
            SetFocus(hwnd);
            return 0;
        }

        if (commandId == IDC_CLEAR) {
            if (history) SetWindowTextW(history, L"");
            return 0;
        }

        if (commandId == IDC_SAVE) {
            const int index =
                static_cast<int>(SendMessageA(combo, CB_GETCURSEL, 0, 0));

            if (index >= 0 &&
                index < static_cast<int>(sizeof(kLanguages) / sizeof(kLanguages[0]))) {
                SaveLanguageCode(kLanguages[index].code);
                SetWindowTextA(status, "Reloading...");
            }
            return 0;
        }

        break;
    }

    case WM_CLOSE:
        KillTimer(hwnd, HOTKEY_CAPTURE_TIMER);
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        ExitProcess(0);
        return 0;
    }

    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

static void SettingsWindowThread() {
    HINSTANCE hInst = GetModuleHandleA(nullptr);

    WNDCLASSA wc{};
    wc.lpfnWndProc = SettingsWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "SpeechHelperSettingsWindow";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.hIcon = LoadIconA(hInst, MAKEINTRESOURCEA(101));

    RegisterClassA(&wc);

    HWND hwnd = CreateWindowExA(
        0,
        wc.lpszClassName,
        "SpeechHelper",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        620, 600,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd) return;

    SendMessageA(hwnd, WM_SETICON, ICON_BIG,
        reinterpret_cast<LPARAM>(LoadIconA(hInst, MAKEINTRESOURCEA(101))));
    SendMessageA(hwnd, WM_SETICON, ICON_SMALL,
        reinterpret_cast<LPARAM>(LoadIconA(hInst, MAKEINTRESOURCEA(101))));

    g_mainWindow = hwnd;
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessageA(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    SetConsoleOutputCP(CP_UTF8);

    g_hotkeyMods.store(LoadHotkeyMods());
    g_hotkeyVk.store(LoadHotkeyVk());

    std::thread settingsThread(SettingsWindowThread);
    settingsThread.detach();

    const fs::path base = ExeDir();

    const fs::path encoder =
        base / "models" / "small-encoder.int8.onnx";

    const fs::path decoder =
        base / "models" / "small-decoder.int8.onnx";

    const fs::path tokens =
        base / "models" / "small-tokens.txt";

    std::cout << "SpeechHelper v4-small\n";
    std::cout << "Portable / offline prototype\n";
    std::cout << "sherpa-onnx: " << SherpaOnnxGetVersionStr() << "\n";
    std::cout << "Talk: "
              << HotkeyBindingName(g_hotkeyMods.load(), g_hotkeyVk.load())
              << "\n";
    std::cout << "ESC: exit\n\n";

    if (!fs::exists(encoder) ||
        !fs::exists(decoder) ||
        !fs::exists(tokens)) {
        std::cerr
            << "[ERROR] Whisper Small INT8 model files are missing.\n"
            << "Expected:\n"
            << "  models\\small-encoder.int8.onnx\n"
            << "  models\\small-decoder.int8.onnx\n"
            << "  models\\small-tokens.txt\n";
        return 1;
    }

    {
        std::lock_guard<std::mutex> lock(g_languageMutex);
        g_requestedLanguage = LoadLanguageCode();
    }

    SpeechRecognizer stt;

    std::cout << "[STT] loading model...\n";
    if (!stt.Init(encoder, decoder, tokens)) {
        return 2;
    }

    std::cout << "[STT] ready\n\n";

    AudioCapture capture;

    bool captureActive = false;

    while (true) {
        if (g_languageReloadRequested.exchange(false)) {
            std::string newLanguage;
            {
                std::lock_guard<std::mutex> lock(g_languageMutex);
                newLanguage = g_requestedLanguage;
            }

            PostUiStatus("Reloading...");

            if (stt.ReloadLanguage(newLanguage)) {
                PostUiStatus("Ready");
            } else {
                PostUiStatus("Reload failed");
            }
        }

        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            break;
        }

        if (g_capturingHotkey.load()) {
            if (captureActive) {
                capture.Stop();
                captureActive = false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            continue;
        }

        if (!captureActive) {
            if (IsSpeechHotkeyDown()) {
                if (capture.Start()) {
                    captureActive = true;
                    PostUiStatus("Listening");
                    std::cout << "[HOTKEY] recording\n";
                }
            }
        } else if (!IsSpeechHotkeyDown()) {
            auto samples = capture.Stop();
            captureActive = false;

            const double seconds =
                static_cast<double>(samples.size()) /
                static_cast<double>(kSampleRate);

            PostUiStatus("Transcribing");
            std::cout
                << "[HOTKEY] captured "
                << seconds
                << " sec\n";

            float rms = 0.0f;
            if (!samples.empty()) {
                double sumSq = 0.0;
                for (float s : samples) {
                    sumSq += static_cast<double>(s) * static_cast<double>(s);
                }
                rms = static_cast<float>(std::sqrt(sumSq / samples.size()));
            }

            if (seconds < 0.25) {
                PostUiStatus("Ready");
                std::cout << "[HOTKEY] ignored: too short\n\n";
            } else if (rms < 0.0035f) {
                PostUiStatus("Ready");
                std::cout << "[HOTKEY] ignored: silence/noise floor"
                          << " (rms=" << rms << ")\n\n";
            } else {
                const auto startTime =
                    std::chrono::steady_clock::now();

                std::string recognized = stt.Transcribe(samples);

                const auto endTime =
                    std::chrono::steady_clock::now();

                const double elapsed =
                    std::chrono::duration<double>(endTime - startTime).count();

                std::cout << "[STT] " << elapsed << " sec\n";

                if (recognized.empty()) {
                    PostUiStatus("Ready");
                    std::cout << "[TEXT] <empty>\n\n";
                } else if (IsNonSpeechLabel(recognized)) {
                    PostUiStatus("Ready");
                    std::cout << "[STT] ignored non-speech label: "
                              << recognized << "\n\n";
                } else {
                    PostUiText(recognized);

                    const bool typed =
                        SendTextToFocusedInput(recognized);

                    PostUiStatus(typed ? "Ready" : "Input send failed");
                    std::cout << "[TEXT] " << recognized << "\n";
                    std::cout << "[INPUT] "
                              << (typed ? "sent" : "send failed")
                              << "\n\n";
                }
            }
        }

        // Idle CPU stays tiny.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(8)
        );
    }

    std::cout << "Bye.\n";
    return 0;
}
