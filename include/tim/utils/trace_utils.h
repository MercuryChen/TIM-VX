#ifndef TIM_VX_UTILS_TRACE_UTILS_H_
#define TIM_VX_UTILS_TRACE_UTILS_H_
#include <memory>
#include <iostream>
#include <functional>
#include <string>
#include <unordered_map>
#include <stdio.h>
#include <stdarg.h>
#include <boost/preprocessor.hpp>
#include <boost/type_index.hpp>

#include "tim/vx/context.h"
#include "context_private.h"
#include "graph_private.h"
#include "tim/vx/compile_option.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"
#include "tim/vx/operation.h"
#include "builtin_op_impl.h"

/************************************************************ 
Caution! Do not formatting these code with auto format tools!
*************************************************************/
namespace trace {
namespace target = ::tim::vx;

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
      VSILOGE("Can not open file at: %s", path);
    }
    return fp;
  }
  template <typename T>
  static std::vector<T>& get_vector(uint32_t offset, size_t vec_size) {
    std::vector<T> ret_vec;
    if (!file_trace_bin) {
      VSILOGE("FILE pointer is NULL");
    } else {
      T* buffer = new T[vec_size];
      fseek(file_trace_bin, offset, SEEK_SET);
      if (fread(buffer, sizeof(T), vec_size, file_trace_bin) == vec_size) {
        ret_vec.assign(buffer, buffer + vec_size);
      } else {
        VSILOGE("read file data failed");
      }
      delete[] buffer;
    }
    return ret_vec;
  }
};
FILE* _VSI_Replayer::file_trace_bin = _VSI_Replayer::open_file("trace_bin.bin");

struct Tensor;
struct _VSI_Tracer {
  static std::unordered_map<const void*, std::string> obj_names_;
  static FILE* file_trace_log;
  static FILE* file_trace_bin;
  static std::vector<std::string> msg_cache_;
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
      VSILOGE("Can not open file at: %s", path);
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
    printf("%s", arg_buffer);
  }
  static void init_msg_cache(uint32_t params_size) {
    msg_cache_.clear();
    msg_cache_.resize(params_size);
    auto size = msg_cache_.size();
    std::cout << "######### size = " << size << std::endl;
  }
  static void insert_msg_cache(std::string param_log, uint32_t idx) {
    msg_cache_[idx] = param_log;
  }
  static void pop_msg_cache() {
    for (uint32_t i; i < msg_cache_.size() - 1; i++) {
      logging_msg("%s,", msg_cache_[i].c_str());
    }
    logging_msg(msg_cache_.back().c_str());
  }
  static uint32_t dump_data(const void* data, size_t byte_size, size_t count) {
    fwrite(data, byte_size, count, file_trace_bin);
    static uint32_t offset = 0;
    offset += byte_size * count;
    return offset;
  }

  template <typename T>
  static void logging_vector(const T& vec, uint32_t idx) {
    uint32_t offset = dump_data(vec.data(), sizeof(vec[0]), vec.size());
    std::string element_type =
        boost::typeindex::type_id_with_cvr<decltype(vec[0])>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "_VSI_Replayer::get_vector<%s>(%u, %u)",
             element_type.c_str(), offset, (uint32_t)vec.size());
    insert_msg_cache(std::string(log_msg), idx);
  }

  template <typename T>
  static void logging_numeric(const T& numeric, uint32_t idx) {
    insert_msg_cache(std::to_string(numeric), idx);
  }

  template <typename T>
  static void logging_enum(const T& enum_val, uint32_t idx) {
    insert_msg_cache(std::to_string((int)enum_val), idx);
  }

  template <typename T>
  static void logging_obj(const T& obj, uint32_t idx) {
    insert_msg_cache(obj._VSI_TraceGetObjName(), idx);
  }
  template <typename T>
  static void logging_obj_ptr(const T& obj, uint32_t idx) {
    insert_msg_cache(obj->_VSI_TraceGetObjName(), idx);
  }

static std::vector<std::shared_ptr<target::Tensor>> proc_obj_ptr_vec(const std::vector<std::shared_ptr<Tensor>>& vec);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  static void logging_param(const std::any& param, bool comma) {
    logging_msg("unimplemented_dtype_logging");
    if (comma) {
      logging_msg(",");
    }
  }
#pragma GCC diagnostic pop

};
std::unordered_map<const void*, std::string> _VSI_Tracer::obj_names_;
std::vector<std::string> _VSI_Tracer::msg_cache_;
FILE* _VSI_Tracer::file_trace_log = _VSI_Tracer::open_file("trace_log.cc");
FILE* _VSI_Tracer::file_trace_bin = _VSI_Tracer::open_file("trace_bin.bin");


template <typename TargetClass>
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

#define _L_NUMERIC 0  // mark process flag = 0
#define VSI_LOG_NUMERIC_(numeric, idx) \
  _VSI_Tracer::logging_numeric<decltype(numeric)>(numeric, idx);

#define _L_VECTOR 1 // mark process flag = 1
#define VSI_LOG_VECTOR_(vec, idx) \
  _VSI_Tracer::logging_vector<decltype(vec)>(vec, idx);

#define _L_ENUM 2 // mark process flag = 2
#define VSI_LOG_ENUM_(enum_val, idx) \
  _VSI_Tracer::logging_enum<decltype(enum_val)>(enum_val, idx);

#define _L_OBJ  3 // mark process flag = 3
#define VSI_LOG_OBJ_(obj, idx) \
  _VSI_Tracer::logging_obj<decltype(obj)>(obj, idx);

#define _L_OBJ  3 // mark process flag = 3
#define VSI_LOG_OBJ_(obj, idx) \
  _VSI_Tracer::logging_obj<decltype(obj)>(obj, idx);

#define _L_OBJ_PTR  4 // mark process flag = 4
#define VSI_LOG_OBJ_PTR_(obj, idx) \
  _VSI_Tracer::logging_obj_ptr<decltype(obj)>(obj, idx);

#define _L_DEFAULT 5  // mark process flag = 4
#define VSI_LOG_DEFAULT_(param, idx) // logging the param in SPECIAL_MACRO_

#define VSI_LOG_PARAM_IMPL_(r, flags, i, param) \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 0), VSI_LOG_NUMERIC_, \
    BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 1), VSI_LOG_VECTOR_, \
      BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 2), VSI_LOG_ENUM_,  \
        BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 3), VSI_LOG_OBJ_,  \
          BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 4), VSI_LOG_OBJ_PTR_, VSI_LOG_DEFAULT_)))))(param, i)

#define VSI_LOG_PARAMS(flags, params)                                       \
  _VSI_Tracer::init_msg_cache(BOOST_PP_SEQ_SIZE(params)); \
  BOOST_PP_SEQ_FOR_EACH_I(VSI_LOG_PARAM_IMPL_, flags, params)


#define _P_DEFAULT 0 // mark process flag = 0
#define VSI_PROCESS_DEFAULT_(param) param

#define _P_TRACED_OBJ 1// mark process flag = 1
#define VSI_PROCESS_OBJ_(traced_obj) traced_obj._VSI_TraceGetImpl()

#define _P_TRACED_OBJ_PTR 2 // mark process flag = 2
#define VSI_PROCESS_OBJ_PTR_(traced_obj_ptr) traced_obj_ptr->_VSI_TraceGetImplSp()

#define _P_TRACED_OBJ_PTR_VEC_ 3 // mark process flag = 3
#define VSI_PROCESS_OBJ_PTR_VEC_(obj_ptrs) _VSI_Tracer::proc_obj_ptr_vec(obj_ptrs)

#define VSI_PROCESS_PARAM_IMPL_COMMA_(r, flags, i, param)                            \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 0), VSI_PROCESS_DEFAULT_, \
    BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 1), VSI_PROCESS_OBJ_, \
      BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(i, flags), 2), VSI_PROCESS_OBJ_PTR_, VSI_PROCESS_OBJ_PTR_VEC_))) \
  (param),

#define VSI_PROCESS_PARAM_IMPL_NO_COMMA_(flag, param)                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(flag, 0), VSI_PROCESS_DEFAULT_, \
    BOOST_PP_IF(BOOST_PP_EQUAL(flag, 1), VSI_PROCESS_OBJ_, \
      BOOST_PP_IF(BOOST_PP_EQUAL(flag, 2), VSI_PROCESS_OBJ_PTR_, VSI_PROCESS_OBJ_PTR_VEC_))) \
  (param)

#define VSI_PROCESS_SINGLE_PARAM_(flags, params) \
  VSI_PROCESS_PARAM_IMPL_NO_COMMA_(BOOST_PP_SEQ_ELEM(0, flags), BOOST_PP_SEQ_ELEM(0, params))

#define VSI_PROCESS_MULTI_PARAMS_(flags, params)                                 \
  BOOST_PP_SEQ_FOR_EACH_I(                                                    \
      VSI_PROCESS_PARAM_IMPL_COMMA_,                                          \
      BOOST_PP_SEQ_SUBSEQ(flags, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(flags))),  \
      BOOST_PP_SEQ_SUBSEQ(params, 0,           \
                          BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params)))) \
  VSI_PROCESS_PARAM_IMPL_NO_COMMA_(                                           \
      BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(flags)), flags),       \
      BOOST_PP_SEQ_ELEM(                                                 \
          BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params)), params))

#define VSI_PROCESS_PARAMS(flags, params)                     \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(flags), 1), \
              VSI_PROCESS_SINGLE_PARAM_, VSI_PROCESS_MULTI_PARAMS_)                 \
  (flags, params)

#define VSI_NAME_A_PARAM_(r, data, i, elem) (param_##i)
#define VSI_ARGS_TYPE_TO_PARAMS(types) \
  BOOST_PP_SEQ_FOR_EACH_I(VSI_NAME_A_PARAM_, _, types)

#define VSI_DECLARE_AN_ARG_COMMA_(r, names, i, type) \
  type BOOST_PP_SEQ_ELEM(i, names),
#define VSI_DECLARE_AN_ARG_NO_COMMA_(name, type) type name

#define VSI_SINGLE_ARG_TYPE_TO_DECLARATION_(type) \
  VSI_DECLARE_AN_ARG_NO_COMMA_(                   \
      BOOST_PP_SEQ_ELEM(0, VSI_ARGS_TYPE_TO_PARAMS(type)), BOOST_PP_SEQ_ELEM(0, type))

#define VSI_MULTI_ARGS_TYPE_TO_DECLARATION_(types)                 \
  BOOST_PP_SEQ_FOR_EACH_I(                                         \
      VSI_DECLARE_AN_ARG_COMMA_,                                   \
      BOOST_PP_SEQ_SUBSEQ(                                         \
          VSI_ARGS_TYPE_TO_PARAMS(types),                          \
          0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types))),              \
      BOOST_PP_SEQ_SUBSEQ(types, 0,                                \
                          BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types)))) \
  VSI_DECLARE_AN_ARG_NO_COMMA_(                                    \
      BOOST_PP_SEQ_ELEM(                                           \
          BOOST_PP_DEC(                                            \
              BOOST_PP_SEQ_SIZE(types)),                           \
          VSI_ARGS_TYPE_TO_PARAMS(types)),                         \
      BOOST_PP_SEQ_ELEM(                                           \
          BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(types)), types))

#define VSI_ARGS_TYPE_TO_DECLARATION(types)                \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(types), 1), \
              VSI_SINGLE_ARG_TYPE_TO_DECLARATION_,         \
              VSI_MULTI_ARGS_TYPE_TO_DECLARATION_)(types)

#define VSI_DEF_MEMFN_SP_2_(ret_class, api_name)     \
  std::shared_ptr<ret_class> api_name() {                 \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                 \
    _VSI_Tracer::logging_msg("auto %s = %s->%s();\n", obj_name.c_str(), \
                              this_obj_name.c_str(), __FUNCTION__);    \
    auto obj = std::make_shared<ret_class>(impl_->api_name());            \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name); \
    return obj;                                                       \
  }

#define VSI_DEF_MEMFN_SP_3_(_1, _2, _3) \
_Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_MEMFN_SP_4_(_1, _2, _3, _4) \
_Pragma("GCC error \"no implementation for 4 args macro overload\"")

#define VSI_DEF_MEMFN_SP_5_(ret_class, api_name, args_type, log_flags, proc_flags) \
  std::shared_ptr<ret_class> api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                    \
    std::string this_obj_name = _VSI_TraceGetObjName();                  \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                    \
    _VSI_Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(),      \
                              this_obj_name.c_str(), __FUNCTION__);       \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                       \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                    \
    auto obj = std::make_shared<ret_class>(                                  \
        impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))));         \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);    \
    return obj;                                                          \
  }

#define VSI_DEF_MEMFN_SP_6_(ret_class, api_name, args_type, log_flags, proc_flags, SPECIAL_MACRO_) \
  std::shared_ptr<ret_class> api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                    \
    std::string this_obj_name = _VSI_TraceGetObjName();                  \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                    \
    _VSI_Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(), \
                              this_obj_name.c_str(), __FUNCTION__);    \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                       \
    SPECIAL_MACRO_(VSI_ARGS_TYPE_TO_PARAMS(args_type))     \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                    \
    auto obj = std::make_shared<ret_class>(                                  \
        impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))));         \
    _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);    \
    return obj;                                                          \
  }


#define VSI_DEF_MEMFN_2_(retval, api_name)        \
  retval api_name() {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    _VSI_Tracer::logging_msg("%s->%s();\n", \
                              this_obj_name.c_str(), __FUNCTION__);    \
    return impl_->api_name();                                         \
  }

#define VSI_DEF_MEMFN_3_(_1, _2, _3) \
_Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_MEMFN_4_(_1, _2, _3, _4) \
_Pragma("GCC error \"no implementation for 4 args macro overload\"")

#define VSI_DEF_MEMFN_5_(retval, api_name, args_type, log_flags, proc_flags) \
  retval api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    _VSI_Tracer::logging_msg("%s->%s(",  \
                              this_obj_name.c_str(), __FUNCTION__);    \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                    \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                 \
    return impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type)));    \
  }

#define VSI_DEF_MEMFN_6_(retval, api_name, args_type, log_flags, proc_flags, \
                         SPECIAL_MACRO_)                        \
  retval api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                    \
    SPECIAL_MACRO_(VSI_ARGS_TYPE_TO_PARAMS(args_type))   \
    _VSI_Tracer::logging_msg(");\n");                                 \
    _VSI_Tracer::pop_msg_cache(); \
    return impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type)));    \
  }

#define VSI_DEF_INPLACE_MEMFN_2_(retval, api_name)        \
  retval api_name() {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    _VSI_Tracer::logging_msg("%s->%s();\n", \
                              this_obj_name.c_str(), __FUNCTION__);    \
    impl_->api_name(); \
    return *this;                                         \
  }

#define VSI_DEF_INPLACE_MEMFN_3_(_1, _2, _3) \
_Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_INPLACE_MEMFN_4_(_1, _2, _3, _4) \
_Pragma("GCC error \"no implementation for 4 args macro overload\"")

#define VSI_DEF_INPLACE_MEMFN_5_(retval, api_name, args_type, log_flags, proc_flags) \
  retval api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    _VSI_Tracer::logging_msg("%s->%s(",  \
                              this_obj_name.c_str(), __FUNCTION__);    \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                    \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                 \
    impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type)));    \
    return *this; \
  }

#define VSI_DEF_INPLACE_MEMFN_6_(retval, api_name, args_type, log_flags, proc_flags, \
                         SPECIAL_MACRO_)                        \
  retval api_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) {                                  \
    std::string this_obj_name = _VSI_TraceGetObjName();               \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                                    \
    SPECIAL_MACRO_(VSI_ARGS_TYPE_TO_PARAMS(args_type))   \
    _VSI_Tracer::logging_msg(");\n");                                 \
    _VSI_Tracer::pop_msg_cache(); \
    impl_->api_name(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type)));    \
    return *this; \
  }

#define VSI_DEF_CONSTRUCTOR_1_(class_name) \
  class_name() { \
    std::string obj_name = _VSI_TraceGetObjName();                          \
    std::string pf(__PRETTY_FUNCTION__);                                       \
    pf.replace(pf.find("trace"), 5, _VSI_TraceApiClassBase::target_namespace); \
    pf.substr(0, pf.find("("));                                                \
    _VSI_Tracer::logging_msg("auto %s = %s();", obj_name.c_str(), pf.c_str()); \
    impl_ = std::make_shared<target::class_name>();                            \
    _VSI_Tracer::insert_obj(static_cast<void*>(impl_.get()), obj_name); \
  }

#define VSI_DEF_CONSTRUCTOR_2_(_1, _2) \
_Pragma("GCC error \"no implementation for 2 args macro overload\"")

#define VSI_DEF_CONSTRUCTOR_3_(_1, _2, _3) \
_Pragma("GCC error \"no implementation for 3 args macro overload\"")

#define VSI_DEF_CONSTRUCTOR_4_(class_name, args_type, log_flags, proc_flags) \
  class_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) { \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                          \
    std::string pf(__PRETTY_FUNCTION__);                                       \
    pf.replace(pf.find("trace"), 5, _VSI_TraceApiClassBase::target_namespace); \
    pf.substr(0, pf.find("("));                                                \
    _VSI_Tracer::logging_msg("auto %s = %s(", obj_name.c_str(), pf.c_str());   \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))  \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_ = std::make_shared<target::class_name>(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))); \
    _VSI_Tracer::insert_obj(static_cast<void*>(impl_.get()), obj_name); \
  }

#define VSI_DEF_CONSTRUCTOR_5_(class_name, args_type, log_flags, proc_flags, SPECIAL_MACRO_) \
  class_name(VSI_ARGS_TYPE_TO_DECLARATION(args_type)) { \
    std::string obj_name = _VSI_Tracer::allocate_obj_name();                          \
    std::string pf(__PRETTY_FUNCTION__);                                       \
    pf.replace(pf.find("trace"), 5, _VSI_TraceApiClassBase::target_namespace); \
    pf.substr(0, pf.find("("));                                                \
    _VSI_Tracer::logging_msg("auto %s = %s(", obj_name.c_str(), pf.c_str());   \
    VSI_LOG_PARAMS(log_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))                      \
    SPECIAL_MACRO_(VSI_ARGS_TYPE_TO_PARAMS(args_type)) \
    _VSI_Tracer::pop_msg_cache(); \
    _VSI_Tracer::logging_msg(");\n");                                          \
    impl_ = std::make_shared<target::class_name>(VSI_PROCESS_PARAMS(proc_flags, VSI_ARGS_TYPE_TO_PARAMS(args_type))); \
    _VSI_Tracer::insert_obj(static_cast<void*>(impl_.get()), obj_name); \
  }

#define LOGGING_PONITER_MSG(offset, length, idx) \
  char log_msg[1024] = {0}; \
  snprintf(log_msg, 1024, "_VSI_Replayer::get_vector<char>(%u, %u).data()", \
            offset, length); \
  _VSI_Tracer::insert_msg_cache(std::string(log_msg), idx);

#define GET_MACRO_OVERLOAD_6_(_1, _2, _3, _4, _5, _6, MACRO, ...) MACRO
#define GET_MACRO_OVERLOAD_5_(_1, _2, _3, _4, _5, MACRO, ...) MACRO

#define VSI_DEF_MEMFN_SP(...) \
  GET_MACRO_OVERLOAD_6_(__VA_ARGS__, \
                      VSI_DEF_MEMFN_SP_6_, \
                      VSI_DEF_MEMFN_SP_5_, \
                      VSI_DEF_MEMFN_SP_4_, \
                      VSI_DEF_MEMFN_SP_3_, \
                      VSI_DEF_MEMFN_SP_2_)(__VA_ARGS__)
#define VSI_DEF_MEMFN(...) \
  GET_MACRO_OVERLOAD_6_(__VA_ARGS__, \
                      VSI_DEF_MEMFN_6_, \
                      VSI_DEF_MEMFN_5_, \
                      VSI_DEF_MEMFN_4_, \
                      VSI_DEF_MEMFN_3_, \
                      VSI_DEF_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_INPLACE_MEMFN(...) \
  GET_MACRO_OVERLOAD_6_(__VA_ARGS__, \
                      VSI_DEF_INPLACE_MEMFN_6_, \
                      VSI_DEF_INPLACE_MEMFN_5_, \
                      VSI_DEF_INPLACE_MEMFN_4_, \
                      VSI_DEF_INPLACE_MEMFN_3_, \
                      VSI_DEF_INPLACE_MEMFN_2_)(__VA_ARGS__)

#define VSI_DEF_CONSTRUCTOR(...) \
  GET_MACRO_OVERLOAD_5_(__VA_ARGS__, \
                      VSI_DEF_CONSTRUCTOR_5_, \
                      VSI_DEF_CONSTRUCTOR_4_, \
                      VSI_DEF_CONSTRUCTOR_3_, \
                      VSI_DEF_CONSTRUCTOR_2_, \
                      VSI_DEF_CONSTRUCTOR_1_)(__VA_ARGS__)

namespace trace {
using ShapeType = std::vector<uint32_t>;

struct TensorSpec : public _VSI_TraceApiClassBase<target::TensorSpec> {
  VSI_DEF_CONSTRUCTOR(TensorSpec)
  VSI_DEF_CONSTRUCTOR(TensorSpec,
                      (target::DataType)(const ShapeType&)(target::TensorAttribute),
                      (_L_ENUM)(_L_VECTOR)(_L_ENUM),
                      (_P_DEFAULT)(_P_DEFAULT)(_P_DEFAULT))
};

} // // namespace trace

namespace trace {
struct Tensor : public _VSI_TraceApiClassBase<target::Tensor> {
  Tensor(const std::shared_ptr<target::Tensor>& impl) { impl_ = impl; }

#define SPECIAL_MACRO_(params) \
  uint32_t data_length = BOOST_PP_SEQ_ELEM(1, params); \
  uint32_t offset = _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params), 1, data_length); \
  LOGGING_PONITER_MSG(offset, data_length, 0)

VSI_DEF_MEMFN(bool,
              CopyDataToTensor,
              (const void*)(uint32_t),
              (_L_DEFAULT)(_L_NUMERIC),
              (_P_DEFAULT)(_P_DEFAULT),
              SPECIAL_MACRO_)
#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params) \
  uint32_t data_length = impl_->GetSpec().GetByteSize(); \
  uint32_t offset = _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params), 1, data_length); \
  LOGGING_PONITER_MSG(offset, data_length, 0)
VSI_DEF_MEMFN(bool,
              CopyDataFromTensor,
              (void*),
              (_L_DEFAULT),
              (_P_DEFAULT),
              SPECIAL_MACRO_)
#undef SPECIAL_MACRO_
};

std::vector<std::shared_ptr<target::Tensor>> _VSI_Tracer::proc_obj_ptr_vec(const std::vector<std::shared_ptr<Tensor>>& vec) {
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
                (_L_OBJ_PTR),
                (_P_TRACED_OBJ_PTR))
  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutput,
                (const std::shared_ptr<Tensor>&),
                (_L_OBJ_PTR),
                (_P_TRACED_OBJ_PTR))
#define SPECIAL_MACRO_(params) \
  uint32_t vec_size = BOOST_PP_SEQ_ELEM(0, params).size(); \
  _VSI_Tracer::logging_msg("{"); \
  for (uint32_t i = 0; i < vec_size - 1; i++) { \
    _VSI_Tracer::logging_msg("%s,", BOOST_PP_SEQ_ELEM(0, params)[i]->_VSI_TraceGetObjName().c_str()); \
  } \
  _VSI_Tracer::logging_msg(BOOST_PP_SEQ_ELEM(0, params).back()->_VSI_TraceGetObjName().c_str()); \
  _VSI_Tracer::logging_msg("}");
  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindInputs,
                (const std::vector<std::shared_ptr<Tensor>>&),
                (_L_DEFAULT),
                (_P_TRACED_OBJ_PTR_VEC_),
                SPECIAL_MACRO_)
  VSI_DEF_INPLACE_MEMFN(Operation&,
                BindOutputs,
                (const std::vector<std::shared_ptr<Tensor>>&),
                (_L_DEFAULT),
                (_P_TRACED_OBJ_PTR_VEC_),
                SPECIAL_MACRO_)
#undef SPECIAL_MACRO_
};

} // namespace trace

// namespace trace {
// struct BuiltinOp : Operation {
//   BuiltinOp(Graph* graph, uint32_t kind, int in_cnt = 0, int out_cnt = 0,
//             DataLayout layout = DataLayout::ANY) {
//   impl_ = std::make_unique<target::BuiltinOpImpl>(graph, kind, in_cnt, out_cnt, layout);
//   }
// };
// } // namespace trace

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

#define SPECIAL_MACRO_(params) \
  if (BOOST_PP_SEQ_ELEM(1, params) == nullptr) { \
    _VSI_Tracer::insert_msg_cache("nullptr", 1); \
  } else { \
    uint32_t data_length = BOOST_PP_SEQ_ELEM(0, params)._VSI_TraceGetImpl().GetByteSize();\
    uint32_t offset = _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(1, params), 1, data_length); \
    LOGGING_PONITER_MSG(offset, data_length, 0) \
  }
  VSI_DEF_MEMFN_SP(Tensor,
                   CreateTensor,
                   (const TensorSpec&)(const void*),
                   (_L_OBJ)(_L_DEFAULT),
                   (_P_TRACED_OBJ)(_P_DEFAULT),
                   SPECIAL_MACRO_)
#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params) \
  if (BOOST_PP_SEQ_ELEM(0, params) == nullptr) { \
    _VSI_Tracer::insert_msg_cache("nullptr", 1); \
  } else { \
    uint32_t data_length = *BOOST_PP_SEQ_ELEM(1, params); \
    uint32_t offset = _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params), 1, data_length); \
    LOGGING_PONITER_MSG(offset, data_length, 0) \
  } \
  uint32_t offset = _VSI_Tracer::dump_data(BOOST_PP_SEQ_ELEM(1, params), sizeof(size_t), 1); \
  char log_msg[1024] = {0}; \
  snprintf(log_msg, 1024, "_VSI_Replayer::get_vector<size_t>(%u, %u).data()", \
            offset, (uint32_t)sizeof(size_t)); \
  _VSI_Tracer::insert_msg_cache(std::string(log_msg), 1);
  VSI_DEF_MEMFN(bool,
                CompileToBinary,
                (void*)(size_t*),
                (_L_DEFAULT)(_L_DEFAULT),
                (_P_DEFAULT)(_P_DEFAULT),
                SPECIAL_MACRO_)
#undef SPECIAL_MACRO_

  VSI_DEF_MEMFN(bool, Compile)
  VSI_DEF_MEMFN(bool, Run)
  template <typename OpType, typename... Params>
  std::shared_ptr<OpType> CreateOperation(Params... parameters) {
    auto op = std::make_shared<OpType>(impl_->CreateOperation<OpType>(parameters...));
    // op_vector_.push_back(op);
    return op;
  }
//  protected:
//   std::vector<std::shared_ptr<Operation>> op_vector_;
};
template <>
std::shared_ptr<trace::ops::Add> Graph::CreateOperation() {
  auto op = std::make_shared<trace::ops::Add>(impl_->CreateOperation<target::ops::Add>());
  // op_vector_.push_back(op);
  return op;
}
template <>
std::shared_ptr<trace::ops::NBG> Graph::CreateOperation(const char* binary, size_t input_count, size_t output_count) {
  auto op = std::make_shared<trace::ops::NBG>(impl_->CreateOperation<target::ops::NBG>(binary, input_count, output_count));
  // op_vector_.push_back(op);
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
  std::string obj_name = _VSI_Tracer::allocate_obj_name();
  std::string pf(__PRETTY_FUNCTION__);
  char log_msg[1024] = {0};
  snprintf(log_msg, 1024, "auto %s =%s;\n", obj_name.c_str(), pf.substr(pf.rfind(" "), pf.size()).c_str());
  _VSI_Tracer::logging_msg(log_msg);
  auto obj = std::make_shared<Context>(target::Context::Create());
  _VSI_Tracer::insert_obj(static_cast<void*>(obj.get()), obj_name);
  return obj;
}

}  // namespace trace
#endif  // TIM_VX_UTILS_TRACE_UTILS_H_
