//A singleton class that handles all GLFW related functions
#include <iostream>
#include <string>

#include "GL/gl.h"
#include "GLFW/glfw3.h"
#include "owl/common/math/vec.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "owl/helper/cuda.h"


class GLFWHandler
{
    public:
        static GLFWHandler* getInstance();
        GLFWwindow* getWindow();
        void initWindow(int width, int height, std::string title);
        void destroyWindow();
        void swapBuffers();
        void pollEvents();
        int windowShouldClose();
        void* getWindowUserPointer();
        void draw(const void* fbpointer);


        struct MouseState {
            owl::vec2f mousePos;
            owl::vec2f mouseDelta;
            bool leftButtonDown;
            bool rightButtonDown;
            bool middleButtonDown;
        } mouseState;
    private:
        GLFWwindow* window;
        owl::vec2i winSize;

        GLuint   fbTexture  {0};
        cudaGraphicsResource_t cuDisplayTexture { 0 };

        static GLFWHandler* instance;
        GLFWHandler();
        ~GLFWHandler();


        void SetCallbacks();
        
};