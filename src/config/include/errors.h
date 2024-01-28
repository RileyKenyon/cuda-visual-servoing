#ifndef IS_ERRORS_H_
#define IS_ERRORS_H_

#ifdef __cplusplus
namespace is_errors {
const int16_t NO_ERR = (0);
const int16_t INIT_ERR = (-1);
const int16_t ARG_ERR = (-2);
} // namespace is_errors

#else

#define NO_ERR (0)
#define INIT_ERR (-1)
#define ARG_ERR (-2)

#endif

#endif // IS_ERRORS_H