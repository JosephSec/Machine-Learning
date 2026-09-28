#pragma once

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <random.h>

#include <debug.h>


typedef struct {
  unsigned int rows;
  unsigned int cols;

  float *data;
} Matrix;

#define EMPTY_MATRIX (Matrix){.rows = 0, .cols = 0, .data = NULL};

inline Matrix create_matrix(unsigned int _rows, unsigned int _cols, float _fillVal) {
  Matrix output;

  const unsigned int elementCount = _rows * _cols;
  output.data = malloc(elementCount * sizeof(float));

  if(output.data != NULL) {
    for(int i = 0; i < elementCount; i++) output.data[i] = _fillVal;
  }

  output.rows = _rows;
  output.cols = _cols;

  return output;
}
inline Matrix create_uniform_matrix(unsigned int _rows, unsigned int _cols, float _min, float _max) {
  Matrix output;
  
  const unsigned int elementCount = _rows * _cols;
  output.data = malloc(elementCount * sizeof(float));

  if(output.data != NULL) fill_random_uniform(output.data, elementCount, _min, _max);

  output.rows = _rows;
  output.cols = _cols;

  return output;
}
inline Matrix create_list_matrix(unsigned int _rows, unsigned int _cols, const float *_data) {
  Matrix output = create_matrix(_rows, _cols, 0);
  memcpy(output.data, _data, _rows * _cols * sizeof(float));
  return output;
}
inline void copy_matrix_inplace(Matrix *_dst, const Matrix *_src) {
  assert(_src->rows > 0 && _src->cols > 0 && "Cant copy matrix when _src->rows or _src->cols == 0");
  
  size_t bufferSize = _src->rows * _src->cols * sizeof(float);
  _dst->data = realloc(_dst->data, bufferSize);

  if(_dst->data != NULL) memcpy(_dst->data, _src->data, bufferSize);

  _dst->rows = _src->rows;
  _dst->cols = _src->cols;
}
inline Matrix copy_matrix(const Matrix *_src) {
  Matrix output;
  copy_matrix_inplace(&output, _src);
  return output;
}


inline void free_matrix(Matrix *_dst) {
  if(_dst->data != NULL) free(_dst->data);
  *_dst = EMPTY_MATRIX;
}


inline float* get_element_ptr_matrix(Matrix *_src, unsigned int _row, unsigned int _col) {
  return _src->data + (_col + _row * _src->cols);
}
inline float get_element_matrix(const Matrix *_src, unsigned int _row, unsigned int _col) {
  return _src->data[_col + _row * _src->cols];
}


inline void multiply_matrices_inplace(Matrix *_dst, const Matrix *_a, const Matrix *_b) {
  ASSERT_MSG(_a->cols == _b->rows, "MATRIX",
    "Can't multiply matrices (%ix%i) and (%ix%i)",
    _a->rows, _a->cols, _b->rows, _b->cols);

  ASSERT_MSG(_dst->data != NULL, "MATRIX",
    "Can't multiply matrices, _dst must be pre-allocated");
    
  ASSERT_MSG(_dst->rows == _a->rows && _dst->cols == _b->cols, "MATRIX",
    "Can't subtract matrices, _dst(%ix%i) matrix shape should match (%ix%i)",
    _dst->rows, _dst->cols, _a->rows, _b->cols);


  for(int i = 0; i < _a->rows; ++i) {
    for(int k = 0; k < _a->cols; ++k) {
      const float aVal = get_element_matrix(_a, i,k);
      
      for(int j = 0; j < _b->cols; ++j) {
        *get_element_ptr_matrix(_dst, i,j) += aVal * get_element_matrix(_b, k,j);
      }
    }
  }
}
inline Matrix multiply_matrices(const Matrix *_a, const Matrix *_b) {
  Matrix output = create_matrix(_a->rows, _b->cols, 0);
  multiply_matrices_inplace(&output, _a, _b);
  return output;
}

inline void add_matrices_inplace(Matrix *_dst, const Matrix *_a, const Matrix *_b) {
  ASSERT_MSG(_a->cols == _b->cols && _a->rows == _b->rows, "MATRIX",
    "Can't add matrices (%ix%i) and (%ix%i)",
    _a->rows, _a->cols, _b->rows, _b->cols);

  ASSERT_MSG(_dst->data != NULL, "MATRIX",
    "Can't add matrices, _dst must be pre-allocated");
    
  ASSERT_MSG(_dst->rows == _a->rows && _dst->cols == _a->cols, "MATRIX",
    "Can't add matrices, _dst(%ix%i) matrix shape should match (%ix%i)",
    _dst->rows, _dst->cols, _a->rows, _a->cols);
  
  const unsigned int elementCount = _a->rows * _a->cols;

  for(int i = 0; i < elementCount; i++) {
    _dst->data[i] = _a->data[i] + _b->data[i];
  }
}
inline Matrix add_matrices(const Matrix *_a, const Matrix *_b) {
  Matrix output = copy_matrix(_a);
  add_matrices_inplace(&output, _a, _b);
  return output;
}

inline void subtract_matrices_inplace(Matrix *_dst, const Matrix *_a, const Matrix *_b) {
  ASSERT_MSG(_a->cols == _b->cols && _a->rows == _b->rows, "MATRIX",
    "Can't subtract matrices (%ix%i) and (%ix%i)",
    _a->rows, _a->cols, _b->rows, _b->cols);

  ASSERT_MSG(_dst->data != NULL, "MATRIX",
    "Can't add matrices, _dst must be pre-allocated");
    
  ASSERT_MSG(_dst->rows == _a->rows && _dst->cols == _a->cols, "MATRIX",
    "Can't subtract matrices, _dst(%ix%i) matrix shape should match (%ix%i)",
    _dst->rows, _dst->cols, _a->rows, _a->cols);

  const unsigned int elementCount = _a->rows * _a->cols;

  for(int i = 0; i < elementCount; i++) {
    _dst->data[i] = _a->data[i] - _b->data[i];
  }
}
inline Matrix subtract_matrices(const Matrix *_a, const Matrix *_b) {
  Matrix output = copy_matrix(_a);
  subtract_matrices_inplace(&output, _a, _b);
  return output;
}


inline void fill_value_matrix(Matrix *_dst, float _value) {
  const unsigned int elementCount = _dst->rows * _dst->cols;
  for(int i = 0; i < elementCount; i++) _dst->data[i] = _value;
}
inline void fill_uniform_matrix(Matrix *_dst, float _min, float _max) {
  fill_random_uniform(_dst->data, _dst->rows * _dst->cols, _min, _max);
}

inline void reshape_matrix(Matrix *_dst, unsigned int _rows, unsigned int _cols) {
  const unsigned int elementCount = _rows * _cols;

  _dst->rows = _rows;
  _dst->cols = _cols;

  if(_dst->data == NULL) _dst->data = malloc(elementCount * sizeof(float));
  else _dst->data = realloc(_dst->data, elementCount * sizeof(float));
}


char* get_string_matrix_row(const float *_src, unsigned int _cols, unsigned int _precision) {
  char *buffer;
  if(_cols == 0) {
    buffer = strdup("[]");
    return buffer;
  }

  size_t sizeBytes = _cols * (_precision + 7) + 2 + 1;
  buffer = malloc(sizeBytes * sizeof(char));

  if(buffer == NULL) return NULL;

  buffer[0] = '[';
  
  int byteIndex = 1;
  for(int i = 0; i < _cols - 1; i++) {
    byteIndex += snprintf(buffer + byteIndex, sizeBytes - byteIndex, "%.*f, ", _precision, _src[i]);
  }
  byteIndex += snprintf(buffer + byteIndex, sizeBytes - byteIndex, "%.*f", _precision, _src[_cols - 1]);
  
  buffer[byteIndex] = ']';
  buffer[byteIndex + 1] = '\0';

  return buffer;
}
inline char* get_string_matrix(const Matrix *_src, unsigned int _precision) {
  const unsigned int elementCount = _src->rows * _src->cols;

  if(elementCount == 0) {
    char *buffer = malloc(3 * sizeof(char));
    if(buffer == NULL) return NULL;

    strcpy(buffer, "[]");
    return buffer;
  }

  size_t sizeBytes = 2 + _src->rows * (1 + (_src->cols * (_precision + 5) + 2 + 1));
  char *buffer = malloc(sizeBytes * sizeof(char));
  
  if(buffer == NULL) return NULL;
  
  strcpy(buffer, "[\n");
  unsigned int byteIndex = 2;
  for(unsigned int row = 0; row < _src->rows - 1; row++) {
    buffer[byteIndex] = '\t';
    byteIndex += 1;

    char *rowStr = get_string_matrix_row(_src->data + (_src->cols * row), _src->cols, _precision);
    strcpy(buffer + byteIndex, rowStr);
    byteIndex += strlen(rowStr);
    free(rowStr);

    buffer[byteIndex] = '\n';
    byteIndex += 1;
  }
  
  buffer[byteIndex] = '\t';
  byteIndex += 1;

  char *rowStr = get_string_matrix_row(_src->data + (_src->cols * (_src->rows - 1)), _src->cols, _precision);
  strcpy(buffer + byteIndex, rowStr);
  byteIndex += strlen(rowStr);
  free(rowStr);

  buffer[byteIndex] = '\n';
  buffer[byteIndex + 1] = ']';
  buffer[byteIndex + 2] = '\0';

  char *trimmed_buffer = realloc(buffer, byteIndex + 3);
  if(trimmed_buffer != NULL) buffer = trimmed_buffer;

  return buffer;
}

void print_matrix(const char *_str, const Matrix *_matrix, unsigned int _precision) {
  char *matrixStr = get_string_matrix(_matrix, 4);
  printf(_str, matrixStr);
  free(matrixStr);
}
void print_matrix_row(const char *_str, const Matrix *_matrix, unsigned int _row, unsigned int _precision) {
  char *rowStr = get_string_matrix_row(_matrix->data + _row * _matrix->cols, _matrix->cols, _precision);
  printf(_str, rowStr);
  free(rowStr);
}
