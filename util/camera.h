#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

class Camera
{
public:
    // constructor and destructor
    Camera(void);
    virtual ~Camera(void);

    // reset
    void Reset(int width, int height);
    //void Lookat(Mesh* mesh);
    
    // get camera matrices:
    inline glm::mat4 GetViewMatrix() {return m_view;}
    inline glm::mat4 GetProjectionMatrix() {return m_projection;}
	inline glm::mat4 GetModelMatrix() {return m_model;}
	
	// get camera position and raycast direction:
    inline glm::vec3 GetCameraPosition() {return m_position;}
    inline float GetCameraDistance() {return m_eye_distance;}
    inline void SetProjectionPlaneDistance(float distance) {m_cached_projection_plane_distance = distance;}
    glm::vec3 GetRaycastDirection(int x, int y);
    glm::vec3 GetCurrentTargetPoint(int x, int y);

    // mouse interactions
    void MouseChangeDistance(float coe, float dx, float dy);
    void MouseChangeLookat(float coe, float dx, float dy);
    void MouseChangeHeadPitch(float coe, float dx, float dy);

	//key interactions
	void KeyChangeScale(bool is_enlarge);
	void KeyChangeTranslate(int dir, bool is_add);
	
    // Draw axis
   // void DrawAxis();

    // resize
    void ResizeWindow(int w, int h);

protected:
    int m_width;
    int m_height;
    float m_znear;
    float m_zfar;
    float m_fovy;

    float m_eye_distance;
    float m_head;
    float m_pitch;

    glm::vec3 m_position;
    glm::vec3 m_up;
    glm::vec3 m_lookat;
    glm::vec3 m_cached_projection_plane_center;
    glm::vec3 m_cached_projection_plane_xdir;
    glm::vec3 m_cached_projection_plane_ydir;
    float m_cached_projection_plane_distance;

    glm::mat4 m_view;
    glm::mat4 m_projection;
	glm::vec3 m_translate;
	glm::vec3 m_scale;
	glm::mat4 m_model;
private:
    // update camera matrices:
    void updateViewMatrix();
    void updateProjectionMatrix();
	void updateModelMatrix();
};

#endif