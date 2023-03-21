#pragma once
#include "owl/owl.h"
#include "Camera.h"

/*Adapted from OWL Samples
 * by Alper Sahistan 03/17/2023
 */

namespace deltaVis
{
class CameraManipulator
{
public:
    CameraManipulator(Camera *camera);
    ~CameraManipulator();

    bool ProcessEvents();

private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    void rotate(const float deg_x, const float deg_y);

    /*! move forward/backward */
    void move(const float step);

    /*! strafe in camera plane */
    void strafe(const owl::vec2f delta);

    /*! mouse got dragged with left button pressedn, by 'delta'
        pixels, at last position where */
    void mouseDragLeft(const owl::vec2i &where, const owl::vec2i &delta);

    /*! mouse got dragged with left button pressedn, by 'delta'
        pixels, at last position where */
    void mouseDragRight(const owl::vec2i &where, const owl::vec2i &delta);

    /*! mouse got dragged with left button pressedn, by 'delta'
        pixels, at last position where */
    void mouseDragCenter(const owl::vec2i &where, const owl::vec2i &delta);

    Camera* camera;
};
}