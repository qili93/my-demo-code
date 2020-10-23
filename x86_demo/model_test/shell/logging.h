#pragma once

#include <time.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdarg.h>

#if !defined(_WIN32)
#include <sys/time.h>
#include <algorithm>  // std::accumulate
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
extern struct timeval;
static int gettimeofday(struct timeval *tp, void *tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;

  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;

  return (0);
}
#endif  // !_WIN32


#define LOG(status) LOG_##status.stream()
#define LOG_INFO LogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define LOG_FATAL LogMessageFatal(__FILE__, __FUNCTION__, __LINE__)

// VLOG()
#define VLOG(level) VLogMessage(__FILE__, __FUNCTION__, __LINE__, level).stream()

#define CHECK(x) if (!(x)) LogMessageFatal(__FILE__, __FUNCTION__, __LINE__).stream() << "Check failed: " #x << ": " // NOLINT(*)
#define _CHECK_BINARY(x, cmp, y) CHECK((x cmp y)) << (x) << "!" #cmp << (y) << " " // NOLINT(*)

#define CHECK_EQ(x, y) _CHECK_BINARY(x, ==, y)
#define CHECK_NE(x, y) _CHECK_BINARY(x, !=, y)
#define CHECK_LT(x, y) _CHECK_BINARY(x, <, y)
#define CHECK_LE(x, y) _CHECK_BINARY(x, <=, y)
#define CHECK_GT(x, y) _CHECK_BINARY(x, >, y)
#define CHECK_GE(x, y) _CHECK_BINARY(x, >=, y)

template <typename T>
static std::string to_string(const T& v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

template <typename T>
static std::string data_to_string(const T* data, const int64_t size) {
  std::stringstream ss;
  ss << "{";
  for (int64_t i = 0; i < size - 1; ++i) {
    ss << data[i] << ",";
  }
  ss << data[size - 1] << "}";
  return ss.str();
}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::stringstream ss;
  if (shape.empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ss << shape[i] << ",";
  }
  ss << shape[shape.size() - 1] << "}";
  return ss.str();
}

static std::string string_format(const std::string fmt_str, ...) {
  /* Reserve two times as much as the length of the fmt_str */
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(
        new char[n]); /* Wrap the plain char array into the unique_ptr */
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

inline void gen_log(std::ostream& log_stream_, const char* file, const char* func, 
                    int lineno, const char* level, const int kMaxLen = 40) {
  const int len = strlen(file);

  std::string time_str;
  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
#if !defined(_WIN32)
  localtime_r(&timestamp, &tm_time);
#else
  localtime_s(&tm_time, &timestamp);
#endif

  // print date / time
  log_stream_ << '[' << level << ' ' << std::setw(2) << 1 + tm_time.tm_mon
              << '/' << std::setw(2) << tm_time.tm_mday << ' ' << std::setw(2)
              << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
              << std::setw(2) << tm_time.tm_sec << '.' << std::setw(3)
              << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func
                << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}

// LogMessage
class LogMessage {
 public:
  LogMessage(const char* file,
             const char* func,
             int lineno,
             const char* level = "I") {
    level_ = level;
    gen_log(log_stream_, file, func, lineno, level);
  }

  ~LogMessage() {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  std::string level_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

// LogMessageFatal
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file,
                  const char* func,
                  int lineno,
                  const char* level = "F")
      : LogMessage(file, func, lineno, level) {}

  ~LogMessageFatal() {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }
};

class VLogMessage {
 public:
  VLogMessage(const char* file,
              const char* func,
              int lineno,
              const int32_t level_int = 0) {
    const char* GLOG_v = std::getenv("GLOG_v");
    GLOG_v_int = (GLOG_v && atoi(GLOG_v) > 0) ? atoi(GLOG_v) : 0;
    this->level_int = level_int;
    if (GLOG_v_int < level_int) {
      return;
    }
    const char* level = to_string(level_int).c_str();
    gen_log(log_stream_, file, func, lineno, level);
  }

  ~VLogMessage() {
    if (GLOG_v_int < this->level_int) {
      return;
    }
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  VLogMessage(const VLogMessage&) = delete;
  void operator=(const VLogMessage&) = delete;
};


// read buffer from file
static std::string ReadFile(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    LOG(FATAL) << "Open file: [" << filename << "] failed.";
  }
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch)) buf.put(ch);
  ifile.close();
  return buf.str();
}