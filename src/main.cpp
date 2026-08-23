
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
static HWND resetButton = nullptr;

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
static HHOOK g_keyboardHook = nullptr;

static bool ModifierMaskDown(int mods);

static constexpr UINT_PTR HOTKEY_CAPTURE_TIMER = 1;

static LRESULT CALLBACK KeyboardHookProc(int code, WPARAM wParam, LPARAM lParam) {
    if (code < 0) {
        return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
    }

    const auto* key = reinterpret_cast<const KBDLLHOOKSTRUCT*>(lParam);
    if (!key || (key->flags & LLKHF_INJECTED)) {
        return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
    }

    if (g_capturingHotkey.load()) {
        return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
    }

    const int hotkeyVk = g_hotkeyVk.load();
    if (hotkeyVk == 0 || static_cast<int>(key->vkCode) != hotkeyVk) {
        return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
    }

    const int mods = g_hotkeyMods.load();
    if (!ModifierMaskDown(mods)) {
        return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
    }

    if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) {
        return 1;
    }

    return CallNextHookEx(g_keyboardHook, code, wParam, lParam);
}


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

    if (!ModifierMaskDown(mods)) return false;
    if (key == 0) return mods != HOTKEY_MOD_NONE;

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
                if (value >= 0 && value < 256) return value;
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
    IDC_TAB_LANGUAGE = 1006,
    IDC_TAB_SHORTCUTS = 1007,
    IDC_TAB_ACTIVITY = 1008,
    IDC_NORMAL_CHANGE = 1010,
    IDC_RESET_SHORTCUTS = 1015
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

static void SetHotkeyValueText(HWND control, const char* text) {
    if (!control) return;
    HWND parent = GetParent(control);
    RECT redraw{};
    GetWindowRect(control, &redraw);
    MapWindowPoints(nullptr, parent, reinterpret_cast<POINT*>(&redraw), 2);
    InflateRect(&redraw, 2, 2);
    SetWindowTextA(control, text);
    RedrawWindow(parent, &redraw, nullptr,
        RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
}

static void RefreshHotkeyLabelFromSavedBinding() {
    if (!g_hotkeyValue) return;

    const std::string label = HotkeyBindingName(g_hotkeyMods.load(), g_hotkeyVk.load());
    SetHotkeyValueText(g_hotkeyValue, label.c_str());
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

namespace SettingsUi {
constexpr COLORREF Background = RGB(14, 17, 23);
constexpr COLORREF Surface = RGB(23, 27, 35);
constexpr COLORREF SurfaceRaised = RGB(31, 36, 46);
constexpr COLORREF Border = RGB(49, 56, 70);
constexpr COLORREF Text = RGB(235, 238, 245);
constexpr COLORREF Muted = RGB(147, 156, 174);
constexpr COLORREF Accent = RGB(119, 92, 255);
constexpr COLORREF AccentHover = RGB(137, 113, 255);
constexpr COLORREF Success = RGB(87, 214, 153);

static HBRUSH backgroundBrush = nullptr;
static HBRUSH surfaceBrush = nullptr;
static HBRUSH fieldBrush = nullptr;
static HFONT titleFont = nullptr;
static HFONT headingFont = nullptr;
static HFONT bodyFont = nullptr;
static HFONT smallFont = nullptr;

static HFONT MakeFont(int pixels, int weight) {
    return CreateFontA(-pixels, 0, 0, 0, weight, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
}

static void SetFont(HWND control, HFONT font) {
    SendMessageA(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

static HWND Label(HWND parent, const char* text, int x, int y, int w, int h,
                  HFONT font = nullptr) {
    HWND result = CreateWindowA("STATIC", text, WS_CHILD | WS_VISIBLE,
        x, y, w, h, parent, nullptr, nullptr, nullptr);
    SetFont(result, font ? font : bodyFont);
    return result;
}

static HWND Button(HWND parent, const char* text, int id,
                   int x, int y, int w, int h) {
    HWND result = CreateWindowA("BUTTON", text,
        WS_CHILD | WS_VISIBLE | BS_OWNERDRAW | WS_TABSTOP,
        x, y, w, h, parent,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
    SetFont(result, bodyFont);
    return result;
}

static void FillRoundedRect(HDC dc, const RECT& rect, COLORREF color, int radius) {
    HBRUSH brush = CreateSolidBrush(color);
    HPEN pen = CreatePen(PS_SOLID, 1, color);
    HGDIOBJ oldBrush = SelectObject(dc, brush);
    HGDIOBJ oldPen = SelectObject(dc, pen);
    RoundRect(dc, rect.left, rect.top, rect.right, rect.bottom, radius, radius);
    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(brush);
    DeleteObject(pen);
}

static LRESULT CALLBACK ComboProc(HWND hwnd, UINT msg, WPARAM wParam,
                                  LPARAM lParam, UINT_PTR, DWORD_PTR) {
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd, &ps);
        RECT rect{};
        GetClientRect(hwnd, &rect);
        FillRect(dc, &rect, fieldBrush);

        char value[80]{};
        GetWindowTextA(hwnd, value, static_cast<int>(sizeof(value)));
        RECT textRect = rect;
        textRect.left += 12;
        textRect.right -= 34;
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, Text);
        HFONT oldFont = reinterpret_cast<HFONT>(SelectObject(dc, bodyFont));
        DrawTextA(dc, value, -1, &textRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

        const int centerX = rect.right - 17;
        const int centerY = (rect.bottom - rect.top) / 2;
        HPEN arrowPen = CreatePen(PS_SOLID, 2, Muted);
        HPEN oldPen = reinterpret_cast<HPEN>(SelectObject(dc, arrowPen));
        MoveToEx(dc, centerX - 4, centerY - 2, nullptr);
        LineTo(dc, centerX, centerY + 2);
        LineTo(dc, centerX + 4, centerY - 2);
        SelectObject(dc, oldPen);
        SelectObject(dc, oldFont);
        DeleteObject(arrowPen);
        EndPaint(hwnd, &ps);
        return 0;
    }
    if (msg == WM_NCPAINT) return 0;
    if (msg == WM_NCDESTROY) RemoveWindowSubclass(hwnd, ComboProc, 1);
    return DefSubclassProc(hwnd, msg, wParam, lParam);
}

static void ApplyDarkControlTheme(HWND control) {
    HMODULE theme = LoadLibraryA("uxtheme.dll");
    if (!theme) return;
    using SetWindowThemeFn = HRESULT(WINAPI*)(HWND, LPCWSTR, LPCWSTR);
    auto setWindowTheme = reinterpret_cast<SetWindowThemeFn>(
        GetProcAddress(theme, "SetWindowTheme"));
    if (setWindowTheme) {
        setWindowTheme(control, L"DarkMode_Explorer", nullptr);
    }
    FreeLibrary(theme);
}
}

static LRESULT CALLBACK SettingsWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    static int modifierOnlyCandidate = HOTKEY_MOD_NONE;
    static HWND combo = nullptr;
    static HWND status = nullptr;
    static HWND history = nullptr;
    static HWND clearButton = nullptr;
    static bool captureWaitingForRelease = false;
    static int activeTab = IDC_TAB_ACTIVITY;
    static std::vector<HWND> languageControls;
    static std::vector<HWND> shortcutControls;
    static std::vector<HWND> activityControls;

    const auto showTab = [&](int tabId) {
        activeTab = tabId;
        const auto showControls = [&](const std::vector<HWND>& controls, bool visible) {
            for (HWND control : controls) ShowWindow(control, visible ? SW_SHOW : SW_HIDE);
        };
        showControls(languageControls, tabId == IDC_TAB_LANGUAGE);
        showControls(shortcutControls, tabId == IDC_TAB_SHORTCUTS);
        showControls(activityControls, tabId == IDC_TAB_ACTIVITY);
        InvalidateRect(hwnd, nullptr, TRUE);
    };

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
            RedrawWindow(g_history, nullptr, nullptr,
                RDW_INVALIDATE | RDW_ERASE | RDW_UPDATENOW);
        }
        free(value);
        return 0;
    }
    case WM_APP_STT_STATUS: {
        char* value = reinterpret_cast<char*>(lParam);
        if (g_status && value) {
            SetWindowTextA(g_status, value);
            RECT redraw{};
            GetWindowRect(g_status, &redraw);
            MapWindowPoints(nullptr, hwnd, reinterpret_cast<POINT*>(&redraw), 2);
            InflateRect(&redraw, 3, 2);
            RedrawWindow(hwnd, &redraw, nullptr,
                RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
        }
        free(value);
        return 0;
    }

    case WM_CREATE: {
        using namespace SettingsUi;
        backgroundBrush = CreateSolidBrush(Background);
        surfaceBrush = CreateSolidBrush(Surface);
        fieldBrush = CreateSolidBrush(SurfaceRaised);
        titleFont = MakeFont(22, FW_SEMIBOLD);
        headingFont = MakeFont(16, FW_SEMIBOLD);
        bodyFont = MakeFont(14, FW_NORMAL);
        smallFont = MakeFont(12, FW_NORMAL);

        Label(hwnd, "Hakk0ni Edition", 32, 18, 210, 30, titleFont);
        Label(hwnd, "Speech-to-text for Windows", 32, 48, 340, 20, smallFont);

        Button(hwnd, "Language", IDC_TAB_LANGUAGE, 32, 82, 112, 34);
        Button(hwnd, "Shortcuts", IDC_TAB_SHORTCUTS, 152, 82, 112, 34);
        Button(hwnd, "Activity", IDC_TAB_ACTIVITY, 272, 82, 112, 34);

        languageControls.push_back(Label(hwnd, "Recognition language", 60, 164, 170, 22));

        combo = CreateWindowA("COMBOBOX", "",
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | CBS_OWNERDRAWFIXED |
                CBS_HASSTRINGS | WS_VSCROLL | WS_TABSTOP,
            238, 157, 236, 280,
            hwnd, reinterpret_cast<HMENU>(IDC_LANGUAGE), nullptr, nullptr);
        SetFont(combo, bodyFont);
        SetWindowSubclass(combo, ComboProc, 1, 0);
        SendMessageA(combo, CB_SETITEMHEIGHT, static_cast<WPARAM>(-1), 30);
        SendMessageA(combo, CB_SETITEMHEIGHT, 0, 28);
        languageControls.push_back(combo);

        for (const auto& lang : kLanguages) {
            SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(lang.label));
        }

        const std::string current = LoadLanguageCode();
        SendMessageA(combo, CB_SETCURSEL, FindLanguageIndex(current), 0);

        languageControls.push_back(Button(hwnd, "Apply", IDC_SAVE, 490, 156, 82, 34));

        // Normal speech stays separate.
        shortcutControls.push_back(Label(hwnd, "Talk", 60, 154, 72, 22));

        g_hotkeyValue = CreateWindowA(
            "STATIC",
            HotkeyBindingName(
                g_hotkeyMods.load(),
                g_hotkeyVk.load()
            ).c_str(),
            WS_CHILD | WS_VISIBLE | SS_CENTER,
            132, 151, 190, 26,
            hwnd, nullptr, nullptr, nullptr);
        SetFont(g_hotkeyValue, bodyFont);
        shortcutControls.push_back(g_hotkeyValue);

        shortcutControls.push_back(Button(hwnd, "Change", IDC_NORMAL_CHANGE, 338, 147, 96, 34));

        shortcutControls.push_back(resetButton = Button(hwnd, "Reset shortcuts",
            IDC_RESET_SHORTCUTS, 60, 316, 132, 34));

        activityControls.push_back(Label(hwnd, "Status", 52, 148, 58, 22, smallFont));

        status = CreateWindowA("STATIC", "Ready",
            WS_CHILD | WS_VISIBLE,
            116, 146, 260, 24,
            hwnd, reinterpret_cast<HMENU>(IDC_STATUS), nullptr, nullptr);
        SetFont(status, bodyFont);
        g_status = status;
        activityControls.push_back(status);

        activityControls.push_back(Label(hwnd, "Conversation history", 52, 184, 220, 22));

        RECT rc{};
        GetClientRect(hwnd, &rc);

        history = CreateWindowExW(
            0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_VSCROLL |
            ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | ES_WANTRETURN,
            52, 214,
            (std::max)(300L, rc.right - 104),
            (std::max)(140L, rc.bottom - 288),
            hwnd, reinterpret_cast<HMENU>(IDC_HISTORY), nullptr, nullptr);
        SetFont(history, bodyFont);
        ApplyDarkControlTheme(history);
        g_history = history;
        activityControls.push_back(history);

        activityControls.push_back(clearButton = Button(hwnd, "Clear history", IDC_CLEAR,
            52, (std::max)(374L, rc.bottom - 54), 116, 34));

        showTab(IDC_TAB_ACTIVITY);

        return 0;
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT: {
        using namespace SettingsUi;
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd, &ps);
        RECT client{};
        GetClientRect(hwnd, &client);
        FillRect(dc, &client, backgroundBrush);
        RECT contentCard{32, 128, client.right - 32, client.bottom - 16};
        FillRoundedRect(dc, contentCard, Surface, 16);
        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_CTLCOLORSTATIC: {
        using namespace SettingsUi;
        HDC dc = reinterpret_cast<HDC>(wParam);
        HWND control = reinterpret_cast<HWND>(lParam);
        if (control == history) {
            SetBkMode(dc, OPAQUE);
            SetBkColor(dc, SurfaceRaised);
            SetTextColor(dc, Text);
            return reinterpret_cast<LRESULT>(fieldBrush);
        }
        if (control == status) {
            SetBkMode(dc, OPAQUE);
            SetBkColor(dc, Surface);
            SetTextColor(dc, Success);
            return reinterpret_cast<LRESULT>(surfaceBrush);
        }
        if (control == g_hotkeyValue) {
                SetBkMode(dc, OPAQUE);
                SetBkColor(dc, Surface);
                SetTextColor(dc, Text);
                return reinterpret_cast<LRESULT>(surfaceBrush);
        }
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, Text);
        return reinterpret_cast<LRESULT>(GetStockObject(NULL_BRUSH));
    }

    case WM_CTLCOLOREDIT:
    case WM_CTLCOLORLISTBOX: {
        using namespace SettingsUi;
        HDC dc = reinterpret_cast<HDC>(wParam);
        SetTextColor(dc, Text);
        SetBkColor(dc, SurfaceRaised);
        return reinterpret_cast<LRESULT>(fieldBrush);
    }

    case WM_DRAWITEM: {
        using namespace SettingsUi;
        auto* item = reinterpret_cast<DRAWITEMSTRUCT*>(lParam);
        if (!item) break;
        if (item->CtlType == ODT_COMBOBOX) {
            HBRUSH comboBrush = (item->itemState & ODS_SELECTED)
                ? surfaceBrush : fieldBrush;
            FillRect(item->hDC, &item->rcItem, comboBrush);
            if (item->itemID != static_cast<UINT>(-1)) {
                char text[80]{};
                SendMessageA(item->hwndItem, CB_GETLBTEXT, item->itemID,
                    reinterpret_cast<LPARAM>(text));
                RECT textRect = item->rcItem;
                textRect.left += 12;
                SetBkMode(item->hDC, TRANSPARENT);
                SetTextColor(item->hDC, Text);
                HFONT oldFont = reinterpret_cast<HFONT>(SelectObject(item->hDC, bodyFont));
                DrawTextA(item->hDC, text, -1, &textRect,
                    DT_LEFT | DT_VCENTER | DT_SINGLELINE);
                SelectObject(item->hDC, oldFont);
            }
            return TRUE;
        }
        if (item->CtlType != ODT_BUTTON) break;
        const bool pressed = (item->itemState & ODS_SELECTED) != 0;
        const bool disabled = (item->itemState & ODS_DISABLED) != 0;
        const bool primary = item->CtlID == IDC_SAVE;
        const bool tab = item->CtlID >= IDC_TAB_LANGUAGE && item->CtlID <= IDC_TAB_ACTIVITY;
        const bool selectedTab = tab && static_cast<int>(item->CtlID) == activeTab;
        COLORREF fill = primary ? (pressed ? AccentHover : Accent)
                                : (selectedTab ? Accent : (pressed ? Border : SurfaceRaised));
        const COLORREF behind = tab ? Background : Surface;
        HBRUSH behindBrush = CreateSolidBrush(behind);
        FillRect(item->hDC, &item->rcItem, behindBrush);
        DeleteObject(behindBrush);
        FillRoundedRect(item->hDC, item->rcItem, fill, 10);
        char text[80]{};
        GetWindowTextA(item->hwndItem, text, static_cast<int>(sizeof(text)));
        SetBkMode(item->hDC, TRANSPARENT);
        SetTextColor(item->hDC, disabled ? Muted : Text);
        HFONT oldFont = reinterpret_cast<HFONT>(SelectObject(item->hDC, bodyFont));
        DrawTextA(item->hDC, text, -1, &item->rcItem,
            DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        SelectObject(item->hDC, oldFont);
        if (item->itemState & ODS_FOCUS) {
            RECT focus = item->rcItem;
            InflateRect(&focus, -4, -4);
            DrawFocusRect(item->hDC, &focus);
        }
        return TRUE;
    }

    case WM_SIZE: {
        const int clientW = LOWORD(lParam);
        const int clientH = HIWORD(lParam);

        if (history) {
            MoveWindow(
                history,
                52, 214,
                (std::max)(300, clientW - 104),
                (std::max)(140, clientH - 288),
                TRUE
            );
        }

        if (clearButton) {
            MoveWindow(
                clearButton,
                52, (std::max)(374, clientH - 54),
                116, 34,
                TRUE
            );
        }

        RedrawWindow(hwnd, nullptr, nullptr,
            RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);

        return 0;
    }

    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        if (info) {
            info->ptMinTrackSize.x = 620;
            info->ptMinTrackSize.y = 500;
        }
        return 0;
    }

    case WM_TIMER:
        if (wParam == HOTKEY_CAPTURE_TIMER && g_capturingHotkey.load()) {
            if (captureWaitingForRelease) {
                if (!AnyCaptureKeyDown()) {
                    captureWaitingForRelease = false;
                    SetHotkeyValueText(g_hotkeyValue, "Press shortcut...");
                }
                return 0;
            }

            const int mods = CurrentModifierMask();
            const int key = FindCaptureMainKey();

            if (key == 0) {
                if (mods != HOTKEY_MOD_NONE) {
                    modifierOnlyCandidate = mods;

                    if (g_hotkeyValue) {
                        const std::string preview = HotkeyBindingName(mods, 0);
                        SetHotkeyValueText(g_hotkeyValue, preview.c_str());
                    }

                    return 0;
                }

                if (modifierOnlyCandidate != HOTKEY_MOD_NONE) {
                    SaveHotkeyBinding(modifierOnlyCandidate, 0);

                    modifierOnlyCandidate = HOTKEY_MOD_NONE;
                    g_capturingHotkey.store(false);
                    captureWaitingForRelease = false;
                    KillTimer(hwnd, HOTKEY_CAPTURE_TIMER);
                    return 0;
                }

                return 0;
            }

            // Save only a non-modifier main key.
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
            return 0;
        }
        break;
    case WM_SYSCHAR:
        if (!g_capturingHotkey.load()) {
            return 0;
        }
    break;
    case WM_SYSCOMMAND:
        if (g_capturingHotkey.load() &&
            (wParam & 0xFFF0) == SC_KEYMENU) {
            return 0;
        }
        break;
    case WM_COMMAND: {
        const int commandId = LOWORD(wParam);

        if (commandId >= IDC_TAB_LANGUAGE && commandId <= IDC_TAB_ACTIVITY) {
            showTab(commandId);
            return 0;
        }
        if (commandId == IDC_NORMAL_CHANGE) {
            captureWaitingForRelease = true;
            modifierOnlyCandidate = HOTKEY_MOD_NONE;
            g_capturingHotkey.store(true);

            SetHotkeyValueText(g_hotkeyValue, "Release, then press...");
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

        if (commandId == IDC_RESET_SHORTCUTS) {
            g_hotkeyMods.store(HOTKEY_MOD_NONE);
            g_hotkeyVk.store(VK_LMENU);

            WriteConfig(LoadLanguageCode());
            RefreshHotkeyLabelFromSavedBinding();
            return 0;
        }

        break;
    }

    case WM_CLOSE:
        KillTimer(hwnd, HOTKEY_CAPTURE_TIMER);
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        DeleteObject(SettingsUi::backgroundBrush);
        DeleteObject(SettingsUi::surfaceBrush);
        DeleteObject(SettingsUi::fieldBrush);
        DeleteObject(SettingsUi::titleFont);
        DeleteObject(SettingsUi::headingFont);
        DeleteObject(SettingsUi::bodyFont);
        DeleteObject(SettingsUi::smallFont);
        PostQuitMessage(0);
        ExitProcess(0);
        return 0;
    }



    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

static void SettingsWindowThread() {
    HINSTANCE hInst = GetModuleHandleA(nullptr);

    g_keyboardHook = SetWindowsHookExA(
        WH_KEYBOARD_LL,
        KeyboardHookProc,
        hInst,
        0
    );

    WNDCLASSA wc{};
    wc.lpfnWndProc = SettingsWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "Hakk0niEditionSettingsWindow";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = CreateSolidBrush(SettingsUi::Background);
    wc.hIcon = LoadIconA(hInst, MAKEINTRESOURCEA(101));

    RegisterClassA(&wc);

    HWND hwnd = CreateWindowExA(
        0,
        wc.lpszClassName,
        "Hakk0ni Edition",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        760, 650,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd) return;

    // Ask supported Windows versions for a dark title bar without adding a
    // hard dependency on dwmapi.dll.
    if (HMODULE dwm = LoadLibraryA("dwmapi.dll")) {
        using DwmSetWindowAttributeFn = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
        auto setAttribute = reinterpret_cast<DwmSetWindowAttributeFn>(
            GetProcAddress(dwm, "DwmSetWindowAttribute"));
        if (setAttribute) {
            const BOOL enabled = TRUE;
            setAttribute(hwnd, 20, &enabled, sizeof(enabled));
        }
        FreeLibrary(dwm);
    }

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

    if (g_keyboardHook) {
        UnhookWindowsHookEx(g_keyboardHook);
        g_keyboardHook = nullptr;
    }
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    SetConsoleOutputCP(CP_UTF8);

    // Load persistent bindings before the settings window is created so the
    // GUI never briefly shows the old default while config.txt is being read.
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

    std::cout << "Hakk0ni Edition v4-small\n";
    std::cout << "Portable / offline prototype\n";
    std::cout << "sherpa-onnx: " << SherpaOnnxGetVersionStr() << "\n";
    std::cout << "Talk: " << HotkeyBindingName(g_hotkeyMods.load(), g_hotkeyVk.load()) << "\n";

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


        // While the settings window is capturing a new binding, the normal
        // speech hotkey must be completely disabled. This prevents ALT/CTRL/etc.
        // from starting a recording at the same time as key capture.
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
            std::cout << "[HOTKEY] captured " << seconds << " sec\n";

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
                const auto startTime = std::chrono::steady_clock::now();
                std::string recognized = stt.Transcribe(samples);
                const auto endTime = std::chrono::steady_clock::now();

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

                    const bool typed = SendTextToFocusedInput(recognized);

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