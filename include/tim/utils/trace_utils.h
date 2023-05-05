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
#include <type_traits>

#include "tim/vx/context.h"
#include "context_private.h"
#include "graph_private.h"
#include "tim/vx/compile_option.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"
#include "tim/vx/operation.h"

/************************************************************ 
Caution! Do not formatting these code with auto format tools!
*************************************************************/
/*
ToDo:
1. split trace implements to multi files (DONE)
2. annotating and readme
3. change some members to private
4. logging enum with literal
*/

namespace trace {
namespace target = ::tim::vx;
template <class, class = void>
struct is_fundamental_vector : std::false_type {};

template <class T>
struct is_fundamental_vector<std::vector<T>> {
  static constexpr bool value = std::is_fundamental_v<T>;
};

template <class T>
struct is_fundamental_pointer : std::integral_constant<bool,
    std::is_pointer_v<T> &&
    std::is_fundamental_v<std::remove_pointer_t<T>>> {};

template <class, class = void>
struct is_traced_obj : std::false_type {};

template <class T>
struct is_traced_obj<T,
    std::void_t<decltype(std::declval<T&>()._VSI_TraceGetObjName())>>
  : std::true_type {};

template <class, class = void>
struct is_traced_obj_ptr : std::false_type {};

template <class T>
struct is_traced_obj_ptr<T,
    std::void_t<decltype(std::declval<T&>()->_VSI_TraceGetObjName())>>
  : std::true_type {};

template <class T>
struct is_others : std::integral_constant<bool,
    !is_fundamental_vector<std::decay_t<T>>::value &&
    !std::is_enum<std::decay_t<T>>::value &&
    !std::is_fundamental<std::decay_t<T>>::value &&
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value> {};

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
      VSILOGE("Can not open file at: %s\n", path);
    }
    return fp;
  }
  template <class T>
  static std::vector<T> get_vector(uint32_t offset, size_t vec_size) {
    std::vector<T> ret_vec;
    if (!file_trace_bin) {
      VSILOGE("FILE pointer is NULL!\n");
    } else {
      T* buffer = new T[vec_size];
      fseek(file_trace_bin, offset, SEEK_SET);
      if (fread(buffer, sizeof(T), vec_size, file_trace_bin) == vec_size) {
        ret_vec.assign(buffer, buffer + vec_size);
      } else {
        VSILOGE("Read bin data failed!\n");
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
      VSILOGE("Can not open file at: %s\n", path);
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
      VSILOGE("Can't amend sub_msg, beacuse msg cache is empty!\n");
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

  static void init_params_log_cache(uint32_t params_size) {
    params_log_cache_.clear();
    params_log_cache_.resize(params_size);
  }

  static void insert_params_log_cache(std::string param_log, uint32_t idx) {
    params_log_cache_[idx] = param_log;
  }

  // pop the log of params into msg cache
  static void pop_params_log_cache() {
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      amend_last_msg_cache(params_log_cache_[i] + ", ");
    }
    amend_last_msg_cache(params_log_cache_.back());
  }

  // directly dump the log of params to file
  static void dump_params_log_cache() {
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      logging_msg("%s, ", params_log_cache_[i].c_str());
    }
    logging_msg(params_log_cache_.back().c_str());
  }

  static uint32_t dump_data(const void* data, size_t byte_size, size_t count) {
    if (fwrite(data, byte_size, count, file_trace_bin) != count) {
      VSILOGE("Write trace binary data failed!\n");
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
      typename std::enable_if_t<is_others<T>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    std::string param_type =
        boost::typeindex::type_id<decltype(t)>().pretty_name();
    VSILOGI("default logging_param substitution call, dtype is %s",
        param_type.c_str());
  }
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
    insert_params_log_cache(std::string(log_msg), idx);
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
    insert_params_log_cache(std::string(log_msg), idx);
  }

  // enable if T is fundamental
  template <class T,
      typename std::enable_if_t<
          std::is_fundamental<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(std::to_string(t), idx);
  }

  // enable if T is derive from _VSI_TraceApiClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(t._VSI_TraceGetObjName(), idx);
  }

  // enable if T is shared_ptr point to object which 
  // derive from _VSI_TraceApiClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr<std::decay_t<T>>::value, int> = 0>
  static void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(t->_VSI_TraceGetObjName(), idx);
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
  std::shared_ptr<TargetClass>& _VSI_TraceGetImplSp() { return impl_; }
  std::string& _VSI_TraceGetObjName() const {
    return _VSI_Tracer::obj_names_[static_cast<const void*>(this)];
  }
  static inline const char* target_namespace = "tim::vx";  // since C++ 17
};

}  // namespace trace

#define LOG_PARAM_IMPL_(r, _, i, param)                                        \
  _VSI_Tracer::logging_param<decltype(param)>(param, i);

#define LOG_PARAMS(params)                                                     \
  _VSI_Tracer::init_params_log_cache(BOOST_PP_SEQ_SIZE(params));               \
  BOOST_PP_SEQ_FOR_EACH_I(LOG_PARAM_IMPL_, _, params)

#define _P_DEFAULT 0 // mark process flag = 0
#define PROC_DEFAULT_(param) param

#define _P_OBJ 1// mark process flag = 1
#define PROC_OBJ_(obj) obj._VSI_TraceGetImpl()

#define _P_OBJ_PTR 2 // mark process flag = 2
#define PROC_OBJ_PTR_(obj_ptr) obj_ptr->_VSI_TraceGetImplSp()

#define _P_OBJ_PTR_VEC_ 3 // mark process flag = 3
#define PROC_OBJ_PTR_VEC_(obj_ptrs) _VSI_Tracer::proc_obj_ptr_vec(obj_ptrs)

#define PROC_PARAM_IMPL_COMMA_(r, flags, i, param)                             \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 0), PROC_DEFAULT_,   \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 1), PROC_OBJ_,       \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 2), PROC_OBJ_PTR_,   \
  PROC_OBJ_PTR_VEC_)))(param),

#define PROC_PARAM_IMPL_NO_COMMA_(flag, param)                                 \
  BOOST_PP_IF(BOOST_PP_EQUAL(flag, 0), PROC_DEFAULT_,                          \
  BOOST_PP_IF(BOOST_PP_EQUAL(flag, 1), PROC_OBJ_,                              \
  BOOST_PP_IF(BOOST_PP_EQUAL(flag, 2), PROC_OBJ_PTR_,                          \
  PROC_OBJ_PTR_VEC_)))(param)

#define PROC_SINGLE_PARAM_(flags, params)                                      \
  PROC_PARAM_IMPL_NO_COMMA_(BOOST_PP_SEQ_ELEM(0, flags),                       \
                            BOOST_PP_SEQ_ELEM(0, params))

#define PROC_MULTI_PARAMS_(flags, params)                                      \
  BOOST_PP_SEQ_FOR_EACH_I(                                                     \
    PROC_PARAM_IMPL_COMMA_,                                                    \
    BOOST_PP_SEQ_SUBSEQ(flags,  0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(flags))),    \
    BOOST_PP_SEQ_SUBSEQ(params, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params))))   \
  PROC_PARAM_IMPL_NO_COMMA_(                                                   \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(flags)), flags),          \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params)), params))

#define PROC_PARAMS(flags, params)                                             \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(flags), 1),                     \
              PROC_SINGLE_PARAM_, PROC_MULTI_PARAMS_)(flags, params)

#define NAME_A_PARAM_(r, data, i, elem) (param_##i)

#define ARGS_TYPE_TO_PARAMS(types)                                             \
  BOOST_PP_SEQ_FOR_EACH_I(NAME_A_PARAM_, _, types)

#define DECLARE_AN_ARG_COMMA_(r, names, i, type)                               \
  type BOOST_PP_SEQ_ELEM(i, names),

#define DECLARE_AN_ARG_NO_COMMA_(name, type) type name

#define SINGLE_ARG_TYPE_TO_DECLARATION_(type)                                  \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(0, ARGS_TYPE_TO_PARAMS(type)),                           \
    BOOST_PP_SEQ_ELEM(0, type))

#define MULTI_ARGS_TYPE_TO_DECLARATION_(types)                                 \
  BOOST_PP_SEQ_FOR_EACH_I(DECLARE_AN_ARG_COMMA_,                               \
    BOOST_PP_SEQ_SUBSEQ(ARGS_TYPE_TO_PARAMS(types),                            \
                        0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types))),            \
    BOOST_PP_SEQ_SUBSEQ(types, 0,                                              \
                        BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types))))               \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types)),                  \
                      ARGS_TYPE_TO_PARAMS(types)),                             \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types)), types))

#define ARGS_TYPE_TO_DECLARATION(types)                                        \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(types), 1),                     \
              SINGLE_ARG_TYPE_TO_DECLARATION_,                                 \
              MULTI_ARGS_TYPE_TO_DECLARATION_)(types)


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

#define VSI_DEF_MEMFN_SP_3_(_1, _2, _3)                                        \
  _Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_MEMFN_SP_4_(ret_class, api_name, args_type, proc_flags)        \
  std::shared_ptr<ret_class> api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {   \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                   \
    _VSI_Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(),            \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(                                                       \
            PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type))));         \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
  }

#define VSI_DEF_MEMFN_SP_5_(ret_class, api_name, args_type, proc_flags,        \
                            SPECIAL_MACRO_)                                    \
  std::shared_ptr<ret_class> api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {   \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#ret_class]); \
    _VSI_Tracer::push_back_msg_cache("auto " + obj_name + " = " + this_obj_name\
        + "->" + __FUNCTION__ + "(");                                          \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    SPECIAL_MACRO_(ARGS_TYPE_TO_PARAMS(args_type))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(                                                       \
            PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type))));         \
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

#define VSI_DEF_MEMFN_3_(_1, _2, _3)                                           \
  _Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_MEMFN_4_(retval, api_name, args_type, proc_flags)              \
  retval api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s(",                                        \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    return impl_->api_name(                                                    \
        PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));              \
  }

#define VSI_DEF_MEMFN_5_(retval, api_name, args_type, proc_flags,              \
                         SPECIAL_MACRO_)                                       \
  retval api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::push_back_msg_cache(                                          \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    SPECIAL_MACRO_(ARGS_TYPE_TO_PARAMS(args_type))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    return impl_->api_name(                                                    \
        PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));              \
  }

#define VSI_DEF_INPLACE_MEMFN_2_(retval, api_name)                             \
  retval api_name() {                                                          \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s();\n",                                    \
                              this_obj_name.c_str(), __FUNCTION__);            \
    impl_->api_name();                                                         \
    return *this;                                                              \
  }

#define VSI_DEF_INPLACE_MEMFN_3_(_1, _2, _3)                                   \
  _Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_INPLACE_MEMFN_4_(retval, api_name, args_type, proc_flags)      \
  retval api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::logging_msg("%s->%s(",                                        \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_->api_name(PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));  \
    return *this;                                                              \
  }

#define VSI_DEF_INPLACE_MEMFN_5_(retval, api_name, args_type, proc_flags,      \
                                 SPECIAL_MACRO_)                               \
  retval api_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                       \
    std::string this_obj_name = _VSI_TraceGetObjName();                        \
    _VSI_Tracer::push_back_msg_cache(                                          \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    SPECIAL_MACRO_(ARGS_TYPE_TO_PARAMS(args_type))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    impl_->api_name(PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));  \
    return *this;                                                              \
  }

#define VSI_DEF_CONSTRUCTOR_1_(class_name)                                     \
  class_name() {                                                               \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::logging_msg("auto %s = %s::%s();", obj_name.c_str(),          \
        _VSI_TraceApiClassBase::target_namespace, __FUNCTION__);               \
    impl_ = std::make_shared<target::class_name>();                            \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define VSI_DEF_CONSTRUCTOR_2_(_1, _2)                                         \
  _Pragma("GCC error \"no implementation for 2 args macro overload\"")

#define VSI_DEF_CONSTRUCTOR_3_(class_name, args_type, proc_flags)              \
  class_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                            \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::logging_msg("auto %s = %s::%s(", obj_name.c_str(),            \
        _VSI_TraceApiClassBase::target_namespace, __FUNCTION__);               \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    _VSI_Tracer::dump_params_log_cache();                                      \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));              \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define VSI_DEF_CONSTRUCTOR_4_(class_name, args_type, proc_flags,              \
                               SPECIAL_MACRO_)                                 \
  class_name(ARGS_TYPE_TO_DECLARATION(args_type)) {                            \
    std::string obj_name =                                                     \
        _VSI_Tracer::allocate_obj_name(_VSI_Tracer::objs_prefix_[#class_name]);\
    _VSI_Tracer::push_back_msg_cache(                                          \
        "auto " + obj_name + " = " + _VSI_TraceApiClassBase::target_namespace  \
        + "::" __FUNCTION__ + "(");                                            \
    LOG_PARAMS(ARGS_TYPE_TO_PARAMS(args_type))                                 \
    SPECIAL_MACRO_(ARGS_TYPE_TO_PARAMS(args_type))                             \
    _VSI_Tracer::pop_params_log_cache();                                       \
    _VSI_Tracer::amend_last_msg_cache(");\n");                                 \
    _VSI_Tracer::msg_cache_sync_to_file();                                     \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(proc_flags, ARGS_TYPE_TO_PARAMS(args_type)));              \
    _VSI_Tracer::insert_obj(static_cast<void*>(this), obj_name);               \
  }

#define LOGGING_PONITER_MSG(offset, length, idx)                               \
  char log_msg[1024] = {0};                                                    \
  snprintf(log_msg, 1024,                                                      \
           "trace::_VSI_Replayer::get_vector<char>(%u, %u).data()",            \
           offset, length);                                                    \
  _VSI_Tracer::insert_params_log_cache(std::string(log_msg), idx);

#define GET_MACRO_OVERLOAD_5_(_1, _2, _3, _4, _5, MACRO, ...) MACRO
#define GET_MACRO_OVERLOAD_4_(_1, _2, _3, _4, MACRO, ...) MACRO

#define VSI_DEF_MEMFN_SP(...)                                                  \
  GET_MACRO_OVERLOAD_5_(__VA_ARGS__,                                           \
                        VSI_DEF_MEMFN_SP_5_,                                   \
                        VSI_DEF_MEMFN_SP_4_,                                   \
                        VSI_DEF_MEMFN_SP_3_,                                   \
                        VSI_DEF_MEMFN_SP_2_)(__VA_ARGS__)

#define VSI_DEF_MEMFN(...)                                                     \
  GET_MACRO_OVERLOAD_5_(__VA_ARGS__,                                           \
                        VSI_DEF_MEMFN_5_,                                      \
                        VSI_DEF_MEMFN_4_,                                      \
                        VSI_DEF_MEMFN_3_,                                      \
                        VSI_DEF_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_INPLACE_MEMFN(...)                                             \
  GET_MACRO_OVERLOAD_5_(__VA_ARGS__,                                           \
                        VSI_DEF_INPLACE_MEMFN_5_,                              \
                        VSI_DEF_INPLACE_MEMFN_4_,                              \
                        VSI_DEF_INPLACE_MEMFN_3_,                              \
                        VSI_DEF_INPLACE_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_CONSTRUCTOR(...)                                               \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        VSI_DEF_CONSTRUCTOR_4_,                                \
                        VSI_DEF_CONSTRUCTOR_3_,                                \
                        VSI_DEF_CONSTRUCTOR_2_,                                \
                        VSI_DEF_CONSTRUCTOR_1_)(__VA_ARGS__)

namespace trace {
using ShapeType = std::vector<uint32_t>;

struct TensorSpec : public _VSI_TraceApiClassBase<target::TensorSpec> {
  VSI_DEF_CONSTRUCTOR(TensorSpec)
  VSI_DEF_CONSTRUCTOR(TensorSpec,
                     (target::DataType)(const ShapeType&)(target::TensorAttribute),
                     (_P_DEFAULT)(_P_DEFAULT)(_P_DEFAULT))
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

VSI_DEF_MEMFN(bool,
              CopyDataToTensor,
              (const void*)(uint32_t),
              (_P_DEFAULT)(_P_DEFAULT),
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
              (void*),
              (_P_DEFAULT),
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
                (const std::shared_ptr<Tensor>&),
                (_P_OBJ_PTR))

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutput,
                (const std::shared_ptr<Tensor>&),
                (_P_OBJ_PTR))

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t vec_size = BOOST_PP_SEQ_ELEM(0, params).size();                     \
  _VSI_Tracer::amend_last_msg_cache("{");                                      \
  for (uint32_t i = 0; i < vec_size - 1; i++) {                                \
    _VSI_Tracer::amend_last_msg_cache(                                         \
        BOOST_PP_SEQ_ELEM(0, params)[i]->_VSI_TraceGetObjName() + ",");        \
  }                                                                            \
  _VSI_Tracer::amend_last_msg_cache(                                           \
      BOOST_PP_SEQ_ELEM(0, params).back()->_VSI_TraceGetObjName());            \
  _VSI_Tracer::amend_last_msg_cache("}");

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindInputs,
                (const std::vector<std::shared_ptr<Tensor>>&),
                (_P_OBJ_PTR_VEC_),
                SPECIAL_MACRO_)

  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutputs,
                (const std::vector<std::shared_ptr<Tensor>>&),
                (_P_OBJ_PTR_VEC_),
                SPECIAL_MACRO_)

#undef SPECIAL_MACRO_
};

} // namespace trace

namespace trace {
namespace ops {

struct Add : Operation {
  Add(const std::shared_ptr<target::ops::Add>& impl) : Operation(impl) {}
};

struct NBG : Operation {
  NBG(const std::shared_ptr<target::ops::NBG>& impl) : Operation(impl) {}
};

} // namespace ops
} // namespace trace

namespace trace {

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
                   (const TensorSpec&)(const void*),
                   (_P_OBJ)(_P_DEFAULT),
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
                (void*)(size_t*),
                (_P_DEFAULT)(_P_DEFAULT),
                SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

  VSI_DEF_MEMFN(bool, Compile)

  VSI_DEF_MEMFN(bool, Run)

  template <class OpType, class... Params>
  std::shared_ptr<OpType> CreateOperation(Params... parameters) {
    auto op = std::make_shared<OpType>(
        impl_->CreateOperation<OpType>(parameters...));
    return op;
  }

};
template <>
std::shared_ptr<trace::ops::Add> Graph::CreateOperation() {
  std::string this_obj_name = _VSI_TraceGetObjName();
  std::string obj_name = _VSI_Tracer::allocate_obj_name("add_");
  _VSI_Tracer::logging_msg(
      "auto %s = %s->CreateOperation<tim::vx::ops::Add>();\n",
      obj_name.c_str(), this_obj_name.c_str());
  auto op = std::make_shared<trace::ops::Add>(
      impl_->CreateOperation<target::ops::Add>());
  _VSI_Tracer::insert_obj(static_cast<void*>(op.get()), obj_name);
  return op;
}

template <>
std::shared_ptr<trace::ops::NBG> Graph::CreateOperation(
    const char* binary, size_t input_count, size_t output_count) {
  std::string this_obj_name = _VSI_TraceGetObjName();
  std::string obj_name = _VSI_Tracer::allocate_obj_name("nbg_");
  std::string buf_name = _VSI_Tracer::allocate_obj_name("nbg_buf_vec_");
  _VSI_Tracer::logging_msg(
      "std::vector<char> %s = trace::_VSI_Replayer::get_vector<char>(5324, 5212);\n",
      buf_name.c_str());
  _VSI_Tracer::logging_msg(
      "auto %s = %s->CreateOperation<tim::vx::ops::NBG>(",
      obj_name.c_str(), this_obj_name.c_str());
  _VSI_Tracer::init_params_log_cache(3);
  if (binary == nullptr) {
    _VSI_Tracer::insert_params_log_cache("nullptr", 1);
  } else {
    uint32_t data_length = 5212;
    _VSI_Tracer::dump_data(binary, sizeof(char), data_length);
    // LOGGING_PONITER_MSG(offset, data_length, 0)
    _VSI_Tracer::insert_params_log_cache(buf_name + ".data()", 0);
  }
  _VSI_Tracer::logging_param<size_t>(input_count, 1);
  _VSI_Tracer::logging_param<size_t>(output_count, 2);
  _VSI_Tracer::dump_params_log_cache();
  _VSI_Tracer::logging_msg(");\n");
  auto op = std::make_shared<trace::ops::NBG>(
      impl_->CreateOperation<target::ops::NBG>(
          binary, input_count, output_count));
  _VSI_Tracer::insert_obj(static_cast<void*>(op.get()), obj_name);
  return op;
}

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
  pf.replace(pf.rfind("trace"), 5, _VSI_TraceApiClassBase::target_namespace);
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
