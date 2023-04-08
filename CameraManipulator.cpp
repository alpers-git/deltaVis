#include "CameraManipulator.h"
#include "GLFWHandler.h"
//#include "owl/owl.h"
#include "owl/common/math/vec.h"

/*Adapted from OWL Samples
* by Alper Sahistan 03/17/2023
*/

namespace deltaVis
{
const float kbd_rotate_degrees = 100.f;
const float degrees_per_drag_fraction = 250;
const float pixels_per_move = 90.f;

CameraManipulator::CameraManipulator(Camera *camera)
{
    this->camera = camera;
}

CameraManipulator::~CameraManipulator()
{
}

bool CameraManipulator::ProcessEvents()
{
    bool eventOccured = false;
    auto glfw = GLFWHandler::getInstance();
    if(glfw->mouseState.leftButtonDown)
    {
        mouseDragLeft(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));
        eventOccured = true;
    }

    if(glfw->mouseState.rightButtonDown)
    {
        mouseDragRight(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));
        eventOccured = true;
    }

    if(glfw->mouseState.middleButtonDown)
    {
        mouseDragCenter(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));
        eventOccured = true;
    }
    return eventOccured;
}

void CameraManipulator::rotate(const float deg_u, const float deg_v)
{
    float rad_u = -(float)M_PI / 180.f * deg_u;
    float rad_v = -(float)M_PI / 180.f * deg_v;

    camera->frame =
        owl::linear3f::rotate(camera->frame.vy, rad_u) *
        owl::linear3f::rotate(camera->frame.vx, rad_v) *
        camera->frame;

    if (camera->forceUp)
        camera->forceUpFrame();
}

void CameraManipulator::strafe(const owl::vec2f step)
{
    camera->position = camera->position - step.x * camera->motionSpeed * camera->frame.vx + step.y * camera->motionSpeed * camera->frame.vy;
}

void CameraManipulator::move(const float step)
{
    camera->position = camera->position + step * camera->motionSpeed * camera->frame.vz;
}

void CameraManipulator::mouseDragLeft(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) / owl::vec2f(glfw->getWindowSize());
    rotate(fraction.x * degrees_per_drag_fraction,
           fraction.y * degrees_per_drag_fraction);
}

void CameraManipulator::mouseDragCenter(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) / owl::vec2f(glfw->getWindowSize());
    strafe(fraction * pixels_per_move);
}

void CameraManipulator::mouseDragRight(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) / owl::vec2f(glfw->getWindowSize());
    move(fraction.y * pixels_per_move);
}
}
