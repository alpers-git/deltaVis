//A singleton class that handles all GLFW related functions
#include <iostream>
#include <string>
#include <unordered_map>

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
        owl::vec2i getWindowSize();
        void draw(const void* fbpointer);


        struct MouseState {
            owl::vec2f position;
            owl::vec2f delta;
            bool leftButtonDown = false;
            bool rightButtonDown = false;
            bool middleButtonDown = false;
        } mouseState;

        struct KeyboardState {
            std::unordered_map<int, int> keys;// Keys and their last GLFW action
            bool isDown(int key) {
                if (keys.find(key) == keys.end()) 
                {
                    return false;
                }
                return true; 
            }

            bool isPressed(int key) {
                if (keys.find(key) == keys.end()) 
                {
                    return false;
                }
                return keys[key] == GLFW_PRESS;
            }

            bool isRepeated(int key) {
                if (keys.find(key) == keys.end()) 
                {
                    return false;
                }
                return keys[key] == GLFW_REPEAT;
            }
        } key;
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