#ifndef TIM_VX_UTILS_TRACE_UTILS_H_
#define TIM_VX_UTILS_TRACE_UTILS_H_
#include <memory>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <list>
#include <stdio.h>
#include <stdarg.h>
#include <boost/preprocessor.hpp>
#include <boost/type_index.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/for_each.hpp>
// #include <boost/mp11.hpp>
// #include <boost/mp11/mpl.hpp>
#include <type_traits>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"

/************************************************************ 
Caution! Do not formatting these code with auto format tools!
*************************************************************/
/*
ToDo:
1. split trace implements to multi files
2. annotating and readme
3. change some members to private
4. logging enum with literal (for some reason, can only use c++14, give up)
5. review lvalue and rvalue usage.
*/

#define TCLOGE(fmt, ...)                                                       \
  printf("[ERROR] [%s:%s:%d]" fmt, __FILE__, __FUNCTION__, __LINE__,           \
         ##__VA_ARGS__)

namespace trace {
namespace target = ::tim::vx;
static const char* __trace_target_namespace_ = "tim::vx";

template<typename...>
using void_t = void;

template <class, class = void>
struct is_fundamental_vector : std::false_type {};

template <class T>
struct is_fundamental_vector<std::vector<T>> {
  static constexpr bool value = std::is_fundamental<T>::value;
};

template <class T>
struct is_fundamental_pointer : std::integral_constant<bool,
    std::is_pointer<T>::value &&
    std::is_fundamental<std::remove_pointer_t<T>>::value> {};

template <class, class = void>
struct is_traced_obj : std::false_type {};

template <class T>
struct is_traced_obj<T,
    void_t<decltype(std::declval<T&>()._VSI_TraceGetObjName())>>
  : std::true_type {};

template <class, class = void>
struct is_traced_obj_ptr : std::false_type {};

template <class T>
struct is_traced_obj_ptr<T,
    void_t<decltype(std::declval<T&>()->_VSI_TraceGetObjName())>>
  : std::true_type {};

template <class, class = void>
struct is_traced_obj_ptr_vector : std::false_type {};

template <class T>
struct is_traced_obj_ptr_vector<std::vector<T>> {
  static constexpr bool value = is_traced_obj_ptr<T>::value;
};

template <class T>
struct is_others_log_type : std::integral_constant<bool,
    !is_fundamental_vector<std::decay_t<T>>::value &&
    !std::is_enum<std::decay_t<T>>::value &&
    !std::is_fundamental<std::decay_t<T>>::value &&
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value> {};

template <class T>
struct is_others_proc_type : std::integral_constant<bool,
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value &&
    !is_traced_obj_ptr_vector<std::decay_t<T>>::value> {};

struct _VSI_Replayer {
  static FILE* file_trace_bin;
  static FILE* open_file(const char* file_name) {
    char* prefix = getenv("VSI_TRACE_PREFIX");
    FILE* fp;
    char path[1024] = {0};
    if (prefix != NULL) {
      strcpy(path, prefix);
      strcat(path, file_name);
    } else {
      strcpy(path, file_name);
    }
    fp = fopen(path, "r");
    if (!fp) {
      TCLOGE("Can not open file at: %s\n", path);
    }
    return fp;
  }
  template <class T>
  static std::vector<T> get_vector(uint32_t offset, size_t vec_size) {
    std::vector<T> ret_vec;
    if (!file_trace_bin) {
      TCLOGE("FILE pointer is NULL!\n");
    } else {
      T* buffer = new T[vec_size];
      fseek(file_trace_bin, offset, SEEK_SET);
      if (fread(buffer, sizeof(T), vec_size, file_trace_bin) == vec_size) {
        ret_vec.assign(buffer, buffer + vec_size);
      } else {
        TCLOGE("Read bin data failed!\n");
      }
      delete[] buffer;
    }
    return ret_vec;
  }
};
FILE* _VSI_Replayer::file_trace_bin =
    _VSI_Replayer::open_file("trace_bin_replay.bin");

struct Tensor;
struct _VSI_Tracer {
  static std::unordered_map<const void*, std::string> obj_names_;
  static std::vector<std::string> params_log_cache_;
  static std::list<std::string> msg_cache_;
  static std::unordered_map<std::string, std::string> objs_prefix_;
  static FILE* file_trace_log;
  static FILE* file_trace_bin;

  static std::string allocate_obj_name(const std::string& prefix = "obj_") {
    static std::unordered_map<std::string, uint32_t> objects_counter;
    if (objects_counter.find(prefix) == objects_counter.end()) {
      objects_counter[prefix] = 0;
    }
    return prefix + std::to_string(objects_counter[prefix]++);
  }

  static void insert_obj(const void* obj_ptr, const std::string& obj_name) {
    obj_names_.insert({obj_ptr, obj_name});
  }

  static FILE* open_file(const char* file_name) {
    char* prefix = getenv("VSI_TRACE_PREFIX");
    FILE* fp;
    char path[1024] = {0};
    if (prefix != NULL) {
      strcpy(path, prefix);
      strcat(path, file_name);
    } else {
      strcpy(path, file_name);
    }
    fp = fopen(path, "w");
    if (!fp) {
      TCLOGE("Can not open file at: %s\n", path);
    }
    return fp;
  }

  static void logging_msg(const char* format, ...) {
    char arg_buffer[1024] = {0};
    va_list args;
    va_start(args, format);
    vsnprintf(arg_buffer, 1024, format, args);
    va_end(args);
    fprintf(file_trace_log, "%s", arg_buffer);
    // printf("%s", arg_buffer);
  }

  static void push_back_msg_cache(const std::string& msg) {
    msg_cache_.push_back(msg);
  }

  static void amend_last_msg_cache(const std::string& msg) {
    if (msg_cache_.empty()) {
      TCLOGE("Can't amend sub_msg, beacuse msg cache is empty!\n");
    }
    msg_cache_.back() += msg;
  }

  static void insert_before_last_msg_cache(const std::string& msg) {
    msg_cache_.insert(--msg_cache_.end(), msg);
  }

  static void msg_cache_sync_to_file() {
    while (!msg_cache_.empty()) {
      logging_msg(msg_cache_.front().c_str());
      msg_cache_.pop_front();
    }
  }

  static void clear_params_log_cache() {
    params_log_cache_.clear();
  }

  static void init_params_log_cache(uint32_t params_size) {
    params_log_cache_.clear();
    params_log_cache_.resize(params_size);
  }

  static void append_params_log_cache(std::string param_log) {
    params_log_cache_.push_back(param_log);
  }

  static void insert_params_log_cache(std::string param_log, uint32_t idx) {
    params_log_cache_[idx] = param_log;
  }

  // pop the log of params into msg cache
  static void pop_params_log_cache() {
    if (params_log_cache_.size() == 0)  return;
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      amend_last_msg_cache(params_log_cache_[i] + ", ");
    }
    amend_last_msg_cache(params_log_cache_.back());
  }

  // directly dump the log of params to file
  static void dump_params_log_cache() {
    if (params_log_cache_.size() == 0)  return;
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      logging_msg("%s, ", params_log_cache_[i].c_str());
    }
    logging_msg(params_log_cache_.back().c_str());
  }

  static uint32_t dump_data(const void* data, size_t byte_size, size_t count) {
    if (fwrite(data, byte_size, count, file_trace_bin) != count) {
      TCLOGE("Write trace binary data failed!\n");
    }
    static uint32_t offset = 0;
    uint32_t temp = offset;
    offset += byte_size * count;
    return temp;
  }

  // special process function for tensor vector
  static std::vector<std::shared_ptr<target::Tensor>> proc_obj_ptr_vec(
      const std::vector<std::shared_ptr<Tensor>>& vec);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  // default substitution
  template <class T,
      typename std::enable_if_t<is_others_log_type<T>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {}
#pragma GCC diagnostic pop

  // enable if T is fundamental std::vector
  template <class T,
      typename std::enable_if_t<
          is_fundamental_vector<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    uint32_t offset = dump_data(t.data(), sizeof(t[0]), t.size());
    std::string element_type =
        boost::typeindex::type_id<decltype(t[0])>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "trace::_VSI_Replayer::get_vector<%s>(%u, %u)",
             element_type.c_str(), offset, (uint32_t)t.size());
    if (idx != static_cast<uint32_t>(-1)) {
      insert_params_log_cache(std::string(log_msg), idx);
    } else {
      append_params_log_cache(std::string(log_msg));
    }
  }

  // enable if T is enum
  template <class T,
      typename std::enable_if_t<
          std::is_enum<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    std::string enum_type =
        boost::typeindex::type_id<decltype(t)>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "(%s)%d", enum_type.c_str(), (int)t);
    if (idx != static_cast<uint32_t>(-1)) {
      insert_params_log_cache(std::string(log_msg), idx);
    } else {
      append_params_log_cache(std::string(log_msg));
    }
  }

  // enable if T is fundamental
  template <class T,
      typename std::enable_if_t<
          std::is_fundamental<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    if (idx != static_cast<uint32_t>(-1)) {
      insert_params_log_cache(std::to_string(t), idx);
    } else {
      append_params_log_cache(std::to_string(t));
    }
  }

  // enable if T is derive from _VSI_TraceApiClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    if (idx != static_cast<uint32_t>(-1)) {
      insert_params_log_cache(t._VSI_TraceGetObjName(), idx);
    } else {
      append_params_log_cache(t._VSI_TraceGetObjName());
    }
  }

  // enable if T is shared_ptr point to object which 
  // derive from _VSI_TraceApiClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    if (idx != static_cast<uint32_t>(-1)) {
      insert_params_log_cache(t->_VSI_TraceGetObjName(), idx);
    } else {
      append_params_log_cache(t->_VSI_TraceGetObjName());
    }
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  // default substitution
  template <class T,
      typename std::enable_if_t<is_others_proc_type<T>::value, int> = 0>
  static T&& proc_param(T&& t) {
    return std::forward<T>(t);
  }
#pragma GCC diagnostic pop

  template <class T,
      typename std::enable_if_t<
          is_traced_obj<std::decay_t<T>>::value, int> = 0>
  static decltype(std::declval<T&>()._VSI_TraceGetImpl())&& proc_param(
      T&& t) {
    return std::forward<T>(t)._VSI_TraceGetImpl();
  }

  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr<std::decay_t<T>>::value, int> = 0>
  static decltype(std::declval<T&>()->_VSI_TraceGetImplSp()) proc_param(
     T&& t) {
    return std::forward<T>(t)->_VSI_TraceGetImplSp();
  }

  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr_vector<std::decay_t<T>>::value, int> = 0>
  static std::vector<decltype(std::declval<T&>()[0]->_VSI_TraceGetImplSp())>
      proc_param(T&& t) {
    std::vector<decltype(t[0]->_VSI_TraceGetImplSp())> impl_vec;
    for (auto& x : std::forward<T>(t)) {
      impl_vec.emplace_back(x->_VSI_TraceGetImplSp());
    }
    return impl_vec;
  }

};
std::unordered_map<const void*, std::string> _VSI_Tracer::obj_names_;
std::vector<std::string> _VSI_Tracer::params_log_cache_;
std::list<std::string> _VSI_Tracer::msg_cache_;
FILE* _VSI_Tracer::file_trace_log = _VSI_Tracer::open_file("trace_log.cc");
FILE* _VSI_Tracer::file_trace_bin = _VSI_Tracer::open_file("trace_bin.bin");
std::unordered_map<std::string, std::string> _VSI_Tracer::objs_prefix_ = {
  {"TensorSpec",  "spec_"      },
  {"Tensor",      "tensor_"    },
  {"Graph",       "graph_"     }
};


template <class TargetClass>
struct _VSI_TraceApiClassBase {
  std::shared_ptr<TargetClass> impl_;
  TargetClass& _VSI_TraceGetImpl() const { return *impl_; }
  // temperary return rvalue to prevent compile error
  std::shared_ptr<TargetClass> _VSI_TraceGetImplSp() { return impl_; }
  std::string& _VSI_TraceGetObjName() const {
    return _VSI_Tracer::obj_names_[static_cast<const void*>(this)];
  }
};

}  // namespace trace

#define LOG_PARAM_IMPL_(r, _, i, param)                                        \
  _VSI_Tracer::logging_param(param, i);

#define LOG_PARAMS(params)                                                     \
  _VSI_Tracer::init_params_log_cache(BOOST_PP_SEQ_SIZE(params));               \
  BOOST_PP_SEQ_FOR_EACH_I(LOG_PARAM_IMPL_, _, params)

#define PROC_PARAM_IMPL_COMMA_(r, _, param)                                    \
  _VSI_Tracer::proc_param(param),

#define PROC_PARAM_IMPL_NO_COMMA_(param)                                       \
  _VSI_Tracer::proc_param(param)

#define PROC_SINGLE_PARAM_(params)                                             \
  PROC_PARAM_IMPL_NO_COMMA_(BOOST_PP_SEQ_ELEM(0, params))

#define PROC_MULTI_PARAMS_(params)                                             \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
    PROC_PARAM_IMPL_COMMA_, _,                                                 \
    BOOST_PP_SEQ_SUBSEQ(params, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params))))   \
  PROC_PARAM_IMPL_NO_COMMA_(                                                   \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params)), params))

#define PROC_PARAMS(params)                                                    \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(params), 1),                    \
              PROC_SINGLE_PARAM_, PROC_MULTI_PARAMS_)(params)


#define NAME_A_PARAM_(r, data, i, elem) (param_##i)

#define ARGS_DESC_TO_PARAMS(descs)                                             \
  BOOST_PP_SEQ_FOR_EACH_I(NAME_A_PARAM_, _, descs)

#define IS_WITH_DEFAULT_VAL_(desc)                                             \
  BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(desc), 2)

#define SET_DEFAULT_VAL_(val) = BOOST_PP_SEQ_HEAD(val)

#define DO_NOTHING(x)

#define DECLARE_AN_ARG_COMMA_(r, names, i, desc)                               \
  BOOST_PP_SEQ_HEAD(desc) BOOST_PP_SEQ_ELEM(i, names)                          \
  BOOST_PP_IF(IS_WITH_DEFAULT_VAL_(desc), SET_DEFAULT_VAL_, DO_NOTHING)        \
    (BOOST_PP_SEQ_TAIL(desc)),

#define DECLARE_AN_ARG_NO_COMMA_(name, desc)                                   \
  BOOST_PP_SEQ_HEAD(desc) name                                                 \
  BOOST_PP_IF(IS_WITH_DEFAULT_VAL_(desc), SET_DEFAULT_VAL_, DO_NOTHING)        \
    (BOOST_PP_SEQ_TAIL(desc))

#define SINGLE_ARG_DESC_TO_DECLARATION_(desc)                                  \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(0, ARGS_DESC_TO_PARAMS(desc)),                           \
    BOOST_PP_SEQ_ELEM(0, desc))

#define MULTI_ARGS_DESC_TO_DECLARATION_(descs)                                 \
  BOOST_PP_SEQ_FOR_EACH_I(DECLARE_AN_ARG_COMMA_,                               \
    BOOST_PP_SEQ_SUBSEQ(ARGS_DESC_TO_PARAMS(descs),                            \
                        0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs))),            \
    BOOST_PP_SEQ_SUBSEQ(descs, 0,                                              \
                        BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs))))               \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs)),                  \
                      ARGS_DESC_TO_PARAMS(descs)),                             \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs)), descs))

#define ARGS_DESC_TO_DECLARATION(descs)                                        \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(descs), 1),                     \
              SINGLE_ARG_DESC_TO_DECLARATION_,                                 \
              MULTI_ARGS_DESC_TO_DECLARATION_)(descs)

#define TO_VARIDIC_IMPL_COMMA_(r, _, elem) elem,
#define TO_VARIDIC_IMPL_NO_COMMA_(elem) elem

#define SEQ_TO_VARIDICS(seqs)                                                  \
  BOOST_PP_SEQ_FOR_EACH(TO_VARIDIC_IMPL_COMMA_, _,                             \
      BOOST_PP_SEQ_SUBSEQ(seqs, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs))))     \
  TO_VARIDIC_IMPL_NO_COMMA_(                                                   \
      BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs)), seqs))

#define VSI_DEF_MEMFN_SP_2_(ret_class, api_name)                               \
  std::shared_ptr<ret_class> api_name() {                                      \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#ret_class]); \
    _VSI_Tracer::logging_msg("auto %s = %s->%s();\n", obj_name.c_str(),        \
                              this_obj_name.c_str(), __FUNCTION__);            \
    auto obj = std::make_shared<ret_class>(impl_->api_name());                 \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
  }

#define VSI_DEF_MEMFN_SP_3_(ret_class, api_name, args_desc)                    \
  std::shared_ptr<ret_class> api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {   \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                   \
    _VSI_Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(),            \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(                                                       \
            PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))));                     \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
  }

#define VSI_DEF_MEMFN_SP_4_(ret_class, api_name, args_desc, SPECIAL_MACRO_)    \
  std::shared_ptr<ret_class> api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {   \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#ret_class]); \
    _VSI_Tracer::push_back_msg_cache("auto " + obj_name + " = " + this_obj_name\
        + "->" + __FUNCTION__ + "(");                                          \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(                                                       \
            PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))));                     \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
  }


#define VSI_DEF_MEMFN_2_(retval, api_name)                                     \
  retval api_name() {                                                          \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s();\n",                                    \
                              this_obj_name.c_str(), __FUNCTION__);            \
    return impl_->api_name();                                                  \
  }

#define VSI_DEF_MEMFN_3_(retval, api_name, args_desc)                          \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s(",                                        \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    return impl_->api_name(                                                    \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
  }

#define VSI_DEF_MEMFN_4_(retval, api_name, args_desc, SPECIAL_MACRO_)          \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::push_back_msg_cache(                                          \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    return impl_->api_name(                                                    \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
  }

#define VSI_DEF_INPLACE_MEMFN_2_(retval, api_name)                             \
  retval api_name() {                                                          \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s();\n",                                    \
                              this_obj_name.c_str(), __FUNCTION__);            \
    impl_->api_name();                                                         \
    return *this;                                                              \
  }

#define VSI_DEF_INPLACE_MEMFN_3_(retval, api_name, args_desc)                  \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s(",                                        \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));              \
    return *this;                                                              \
  }

#define VSI_DEF_INPLACE_MEMFN_4_(retval, api_name, args_desc, SPECIAL_MACRO_)  \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::push_back_msg_cache(                                          \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));              \
    return *this;                                                              \
  }

#define VSI_DEF_CONSTRUCTOR_1_(class_name)                                     \
  class_name() {                                                               \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::logging_msg("auto %s = %s::%s();", obj_name.c_str(),          \
        __trace_target_namespace_, __FUNCTION__);                              \
    impl_ = std::make_shared<target::class_name>();                            \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define VSI_DEF_CONSTRUCTOR_2_(class_name, args_desc)                          \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::logging_msg("auto %s = %s::%s(", obj_name.c_str(),            \
        __trace_target_namespace_, __FUNCTION__);                              \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define VSI_DEF_CONSTRUCTOR_3_(class_name, args_desc, SPECIAL_MACRO_)          \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::push_back_msg_cache(                                          \
        "auto " + obj_name + " = " + __trace_target_namespace_ + "::" +        \
        __FUNCTION__ + "(");                                                   \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define SPECIALIZATION_CREATE_OP_1_(opname)                                    \
template <class... Params>                                                     \
inline std::shared_ptr<trace::ops::opname> Graph::CreateOperationImpl(         \
      ops::_VSI_Tag_of_ ## opname, Params... params) {                         \
  std::string this_obj_name = _VSI_TraceGetObjName();                          \
  std::string obj_name = _VSI_Tracer::allocate_obj_name(std::string(#opname) + \
      "_");                                                                    \
  _VSI_Tracer::logging_msg(                                                    \
      "auto %s = %s->CreateOperation<%s::ops::%s>(", obj_name.c_str(),         \
      this_obj_name.c_str(), __trace_target_namespace_, #opname);              \
  _VSI_Tracer::clear_params_log_cache();                                       \
  boost::hana::tuple<Params...> params_tuple = {params...};                    \
  boost::hana::for_each(params_tuple, [&] (auto x) {                           \
    _VSI_Tracer::logging_param(x, -1);                                         \
  });                                                                          \
  _VSI_Tracer::dump_params_log_cache();                                        \
  _VSI_Tracer::logging_msg(");\n");                                            \
  auto op = std::make_shared<trace::ops::opname>(                              \
      impl_->CreateOperation<target::ops::opname>(params...));                 \
  _VSI_Tracer::insert_obj(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define SPECIALIZATION_CREATE_OP_2_(opname, args_desc)                         \
template <>                                                                    \
std::shared_ptr<trace::ops::opname> Graph::CreateOperation(                    \
    ARGS_DESC_TO_DECLARATION(args_desc)) {                                     \
  std::string this_obj_name = _VSI_TraceGetObjName();                          \
  std::string obj_name = _VSI_Tracer::allocate_obj_name(std::string(#opname) + \
      "_");                                                                    \
  _VSI_Tracer::logging_msg(                                                    \
      "auto %s = %s->CreateOperation<target::ops::%s>(",                       \
      obj_name.c_str(), this_obj_name.c_str(), #opname);                       \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
  auto op = std::make_shared<trace::ops::opname>(                              \
      impl_->CreateOperation<target::ops::opname>(                             \
          SEQ_TO_VARIDICS(ARGS_DESC_TO_PARAMS(args_desc))));                   \
  _VSI_Tracer::insert_obj(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define SPECIALIZATION_CREATE_OP_3_(opname, args_desc, SPECIAL_MACRO_)         \
template <>                                                                    \
std::shared_ptr<trace::ops::opname> Graph::CreateOperation(                    \
    ARGS_DESC_TO_DECLARATION(args_desc)) {                                     \
  std::string this_obj_name = _VSI_TraceGetObjName();                          \
  std::string obj_name = _VSI_Tracer::allocate_obj_name(std::string(#opname)   \
      + "_");                                                                  \
  _VSI_Tracer::push_back_msg_cache(                                            \
      "auto " + obj_name + " = " + this_obj_name +                             \
      "->CreateOperation<target::ops::" + #opname + ">(");                     \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
  auto op = std::make_shared<trace::ops::opname>(                              \
      impl_->CreateOperation<target::ops::opname>(                             \
          SEQ_TO_VARIDICS(ARGS_DESC_TO_PARAMS(args_desc))));                   \
  _VSI_Tracer::insert_obj(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define LOGGING_PONITER_MSG(offset, length, idx)                               \
  char log_msg[1024] = {0};                                                    \
  snprintf(log_msg, 1024,                                                      \
           "trace::_VSI_Replayer::get_vector<char>(%u, %u).data()",            \
           offset, length);                                                    \
  _VSI_Tracer::insert_params_log_cache(std::string(log_msg), idx);

#define GET_MACRO_OVERLOAD_4_(_1, _2, _3, _4, MACRO, ...) MACRO
#define GET_MACRO_OVERLOAD_3_(_1, _2, _3, MACRO, ...) MACRO

#define VSI_DEF_MEMFN_SP(...)                                                  \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        VSI_DEF_MEMFN_SP_4_,                                   \
                        VSI_DEF_MEMFN_SP_3_,                                   \
                        VSI_DEF_MEMFN_SP_2_)(__VA_ARGS__)

#define VSI_DEF_MEMFN(...)                                                     \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        VSI_DEF_MEMFN_4_,                                      \
                        VSI_DEF_MEMFN_3_,                                      \
                        VSI_DEF_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_INPLACE_MEMFN(...)                                             \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        VSI_DEF_INPLACE_MEMFN_4_,                              \
                        VSI_DEF_INPLACE_MEMFN_3_,                              \
                        VSI_DEF_INPLACE_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_CONSTRUCTOR(...)                                               \
  GET_MACRO_OVERLOAD_3_(__VA_ARGS__,                                           \
                        VSI_DEF_CONSTRUCTOR_3_,                                \
                        VSI_DEF_CONSTRUCTOR_2_,                                \
                        VSI_DEF_CONSTRUCTOR_1_)(__VA_ARGS__)

#define VSI_SPECIALIZATION_CREATE_OP(...)                                      \
  GET_MACRO_OVERLOAD_3_(__VA_ARGS__,                                           \
                        SPECIALIZATION_CREATE_OP_3_,                           \
                        SPECIALIZATION_CREATE_OP_2_,                           \
                        SPECIALIZATION_CREATE_OP_1_)(__VA_ARGS__)

namespace trace {
using ShapeType = std::vector<uint32_t>;

struct TensorSpec : public _VSI_TraceApiClassBase<target::TensorSpec> {
  VSI_DEF_CONSTRUCTOR(TensorSpec)
  VSI_DEF_CONSTRUCTOR(TensorSpec,
                     ((target::DataType))((const ShapeType&))
                        ((target::TensorAttribute)))
};

} // // namespace trace

namespace trace {
struct Tensor : public _VSI_TraceApiClassBase<target::Tensor> {
  Tensor(const std::shared_ptr<target::Tensor>& impl) { impl_ = impl; }

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t data_length = BOOST_PP_SEQ_ELEM(1, params);                         \
  uint32_t offset =                                                            \
      _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params),                     \
                             sizeof(char), data_length);                       \
  LOGGING_PONITER_MSG(offset, data_length, 0)

// arguments description must format as: ((dtype)) or ((dtype)(default_value))
VSI_DEF_MEMFN(bool,
              CopyDataToTensor,
              ((const void*))((uint32_t)(0)),
              SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t data_length = impl_->GetSpec().GetByteSize();                       \
  uint32_t offset =                                                            \
      _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params),                     \
                             sizeof(char), data_length);                       \
  LOGGING_PONITER_MSG(offset, data_length, 0)

VSI_DEF_MEMFN(bool,
              CopyDataFromTensor,
              ((void*)),
              SPECIAL_MACRO_)

#undef SPECIAL_MACRO_
};

std::vector<std::shared_ptr<target::Tensor>> _VSI_Tracer::proc_obj_ptr_vec(
    const std::vector<std::shared_ptr<Tensor>>& vec) {
  std::vector<std::shared_ptr<target::Tensor>> impl_vec;
  for (auto& x : vec) {
    impl_vec.emplace_back(x->_VSI_TraceGetImplSp());
  }
  return impl_vec;
 }

} // namespace trace

namespace trace {

struct Operation : public _VSI_TraceApiClassBase<target::Operation> {
  Operation(const std::shared_ptr<target::Operation>& impl) { impl_ = impl; }

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindInput,
                ((const std::shared_ptr<Tensor>&)))

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutput,
                ((const std::shared_ptr<Tensor>&)))

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t vec_size = BOOST_PP_SEQ_ELEM(0, params).size();                     \
  _VSI_Tracer::amend_last_msg_cache("{");                                      \
  for (uint32_t i = 0; i < vec_size - 1; i++) {                                \
    _VSI_Tracer::amend_last_msg_cache(                                         \
        BOOST_PP_SEQ_ELEM(0, params)[i]->_VSI_TraceGetObjName() + ", ");       \
  }                                                                            \
  _VSI_Tracer::amend_last_msg_cache(                                           \
      BOOST_PP_SEQ_ELEM(0, params).back()->_VSI_TraceGetObjName());            \
  _VSI_Tracer::amend_last_msg_cache("}");

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindInputs,
                ((const std::vector<std::shared_ptr<Tensor>>&)),
                SPECIAL_MACRO_)

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutputs,
                ((const std::vector<std::shared_ptr<Tensor>>&)),
                SPECIAL_MACRO_)

#undef SPECIAL_MACRO_
};

} // namespace trace

namespace trace {
namespace ops {

struct DefaultTag {};
template<class T>
struct TagDispatchTrait {
  using tag = DefaultTag;
};

#define DEF_TIMVX_OP_IMPL_(r, _, op)                                           \
struct op : Operation {                                                        \
  op(const std::shared_ptr<target::ops::op>& impl) : Operation(impl) {}        \
};                                                                             \
struct BOOST_PP_CAT(_VSI_Tag_of_, op) {};                                      \
template<>                                                                     \
struct TagDispatchTrait<op> {                                                  \
  using tag = BOOST_PP_CAT(_VSI_Tag_of_, op);                                  \
};


#define DEF_TIMVX_OPS_AND_TAGS(ops)                                            \
  BOOST_PP_SEQ_FOR_EACH(DEF_TIMVX_OP_IMPL_, _, ops)

DEF_TIMVX_OPS_AND_TAGS(
  (Add)
  (Reshape)
  (NBG)
)

} // namespace ops
} // namespace trace

namespace trace {

#define DECL_CREATE_OP_IMPL(op)                                                \
  template <class... Params>                                                   \
  inline std::shared_ptr<trace::ops::op> CreateOperationImpl(                  \
      ops::_VSI_Tag_of_ ## op, Params... params);                              \

struct Graph : public _VSI_TraceApiClassBase<target::Graph> {
  Graph(const std::shared_ptr<target::Graph>& impl) { impl_ = impl; }

#define SPECIAL_MACRO_(params)                                                 \
  if (BOOST_PP_SEQ_ELEM(1, params) == nullptr) {                               \
    _VSI_Tracer::insert_params_log_cache("nullptr", 1);                        \
  } else {                                                                     \
    uint32_t data_length =                                                     \
        BOOST_PP_SEQ_ELEM(0, params)._VSI_TraceGetImpl().GetByteSize();        \
    uint32_t offset =                                                          \
        _VSI_Tracer::dump_data(                                                \
            BOOST_PP_SEQ_ELEM(1, params), sizeof(char), data_length);          \
    LOGGING_PONITER_MSG(offset, data_length, 0)                                \
  }

  VSI_DEF_MEMFN_SP(Tensor,
                   CreateTensor,
                   ((const TensorSpec&))((const void*)(nullptr)),
                   SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params)                                                 \
  if (BOOST_PP_SEQ_ELEM(0, params) == nullptr) {                               \
    std::string size_name = _VSI_Tracer::allocate_obj_name("nbg_size_");       \
    _VSI_Tracer::insert_obj(BOOST_PP_SEQ_ELEM(1, params), size_name);          \
    _VSI_Tracer::insert_before_last_msg_cache(                                 \
        "size_t " + size_name + " = -1;\n");                                   \
    _VSI_Tracer::insert_params_log_cache("nullptr", 0);                        \
  } else {                                                                     \
    uint32_t data_length = *BOOST_PP_SEQ_ELEM(1, params);                      \
    uint32_t offset = _VSI_Tracer::dump_data(                                  \
        BOOST_PP_SEQ_ELEM(0, params), sizeof(char), data_length);              \
    LOGGING_PONITER_MSG(offset, data_length, 0)                                \
  }                                                                            \
  _VSI_Tracer::insert_params_log_cache(                                        \
      "&" + _VSI_Tracer::obj_names_[BOOST_PP_SEQ_ELEM(1, params)], 1);

  VSI_DEF_MEMFN(bool,
                CompileToBinary,
                ((void*))((size_t*)),
                SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

  VSI_DEF_MEMFN(bool, Compile)

  VSI_DEF_MEMFN(bool, Run)

  template <class OpType, class... Params>
  std::shared_ptr<OpType> CreateOperation(Params... params) {
    return CreateOperationImpl(
      typename ops::TagDispatchTrait<OpType>::tag {}, params...);
  }

 private:
  DECL_CREATE_OP_IMPL(Add)
  DECL_CREATE_OP_IMPL(Reshape)
};

VSI_SPECIALIZATION_CREATE_OP(Add)
VSI_SPECIALIZATION_CREATE_OP(Reshape)

#define SPECIAL_MACRO_(params)                                                 \
  std::string buf_name = _VSI_Tracer::allocate_obj_name("nbg_buf_vec_");       \
  uint32_t data_length = 5212;                                                 \
  uint32_t offset = _VSI_Tracer::dump_data(                                    \
      BOOST_PP_SEQ_ELEM(0, params), sizeof(char), data_length);                \
  _VSI_Tracer::insert_before_last_msg_cache("std::vector<char> " + buf_name +  \
      " = trace::_VSI_Replayer::get_vector<char>(" + std::to_string(offset)  + \
      "," + std::to_string(data_length) + ");\n");                             \
  _VSI_Tracer::insert_params_log_cache(buf_name + ".data()", 0);

VSI_SPECIALIZATION_CREATE_OP(NBG,
                             ((const char*))((size_t))((size_t)),
                             SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

}  // namespace trace

namespace trace {

struct Context : public _VSI_TraceApiClassBase<target::Context> {
  Context(const std::shared_ptr<target::Context>& impl) { impl_ = impl; }

  static std::shared_ptr<Context> Create();

  VSI_DEF_MEMFN_SP(Graph, CreateGraph)
};
std::shared_ptr<Context> Context::Create() {
  std::string obj_name = _VSI_Tracer::allocate_obj_name("ctx_");
  std::string pf(__PRETTY_FUNCTION__);
  pf.replace(pf.rfind("trace"), 5, __trace_target_namespace_);
  char log_msg[1024] = {0};
  snprintf(log_msg, 1024, "auto %s =%s;\n", obj_name.c_str(),
           pf.substr(pf.rfind(" "), pf.size()).c_str());
  _VSI_Tracer::logging_msg(log_msg);
  auto obj = std::make_shared<Context>(target::Context::Create());
  _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);
  return obj;
}

}  // namespace trace
#endif  // TIM_VX_UTILS_TRACE_UTILS_H_
