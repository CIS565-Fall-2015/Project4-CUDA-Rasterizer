#pragma once

struct Camera {
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec2 fov;
    glm::vec3 light;
};
