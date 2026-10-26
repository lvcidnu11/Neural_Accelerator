#include <cstdarg>
#include <cstdio>
#include <cstdint>
#ifdef __riscv
// Keep formatted diagnostics in RAM; the Spike debugger can inspect them.
extern "C" char last_debug_log[512];
char last_debug_log[512];
extern "C" void DebugLog(const char* format, va_list args) {
  vsnprintf(last_debug_log, sizeof(last_debug_log), format, args);
}
#else
extern "C" void DebugLog(const char* format, va_list args) {
  vfprintf(stderr, format, args);
}
#endif
extern "C" int DebugVsnprintf(char* buffer, size_t size, const char* format, va_list args) {
  return vsnprintf(buffer, size, format, args);
}
#ifdef __riscv
// No heap is supplied. These stubs only satisfy newlib diagnostics dependencies.
extern "C" void* _sbrk(ptrdiff_t) { return reinterpret_cast<void*>(-1); }
extern "C" int _write(int, const void*, unsigned) { return -1; }
extern "C" int _read(int, void*, unsigned) { return -1; }
extern "C" int _close(int) { return -1; }
extern "C" int _fstat(int, void*) { return -1; }
extern "C" int _isatty(int) { return 0; }
extern "C" int _lseek(int, int, int) { return -1; }
extern "C" int _getpid() { return 1; }
extern "C" int _kill(int, int) { return -1; }
extern "C" void _exit(int) { for (;;) __asm__ volatile(""); }
#endif
