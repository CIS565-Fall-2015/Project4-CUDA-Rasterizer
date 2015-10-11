#pragma once

struct Camera {
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    float fovy;
    glm::vec3 light;
};
