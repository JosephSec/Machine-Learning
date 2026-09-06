#include <System/GPUMath.hpp>
#include <System/Debug.hpp>

#include <GL/glew.h>

const char* computeShaderSource = R"(
#version 430 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict buffer BufferA { float a[]; };
layout(std430, binding = 1) readonly buffer BufferB { float b[]; };

uniform int size;
uniform float scalar;

void main() {
  uint idx = gl_GlobalInvocationID.x; 
  
  if(idx < size) a[idx] -= b[idx] * scalar;
}
)";


static GLuint ssboA, ssboB;
static GLuint program;
void GPUMath::Init() {
  { //GLEW
    glewExperimental = GL_TRUE;
    if(GLenum result = glewInit(); result != GLEW_OK) Debug::log("[GLEW ERROR]: glewInit() returned", result);
  }

  { //Buffers
    glGenBuffers(1, &ssboA);
    glGenBuffers(1, &ssboB);
  }

  { //Compute Shader
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    program = glCreateProgram();


    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboB);

    glShaderSource(shader, 1, &computeShaderSource, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
      char infoLog[512];
      glGetShaderInfoLog(shader, 512, NULL, infoLog);
      Debug::log("GLSL Compute shader Compilation Failed", infoLog);
    }

    glAttachShader(program, shader);
    glLinkProgram(program);
    glUseProgram(program);

    glDeleteShader(shader);
  }
}
void GPUMath::ClearGPUMemory() {
  if(ssboA) glDeleteBuffers(1, &ssboA);
  if(ssboB) glDeleteBuffers(1, &ssboB);

  if(program) glDeleteProgram(program);
}

Matrix GPUMath::ASubtractBMulScalar(Matrix& a, Matrix& b, float scalar) {
  int size = static_cast<int>(a.width * a.height);
  size_t bytes = size * sizeof(float);

  glUseProgram(program);

  { //Init Buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, a.GetDataPtr(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, b.GetDataPtr(), GL_DYNAMIC_DRAW);
  }

  glUniform1i(glGetUniformLocation(program, "size"), size);
  glUniform1f(glGetUniformLocation(program, "scalar"), scalar); 
  
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;  
  glDispatchCompute(blocksPerGrid, 1, 1);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  Matrix returnValue = a;
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, returnValue.GetDataPtr());

  glUseProgram(0);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  return returnValue;
}