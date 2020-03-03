#pragma once

#include "scenes/base/base.hpp"
#ifdef SCENE_SPH2

// SPH simulation parameters
struct sph_parameters
{
    float h;         // influence distance of a particle
    // float rho0;      // rest density
    // float m;         // total mass of a particle
    float stiffness; // constant of tait equation (relation density / pression)
    float nu;        // viscosity parameter
    float k; // kernel coef.
    float gradk; // grad kernel coef.
};

// SPH Particle
struct particle_element
{
    vcl::vec3 p; // Position
    vcl::vec3 v; // Speed
    vcl::vec3 a; // Acceleration

    // local density and pression
    float rho;
    float pression;
    float rho0;
    float m;

    // sph_parameters * sph_param;

    particle_element() : p{0, 0, 0}, v{0, 0, 0}, a{0, 0, 0}, rho(0), pression(0) {}

};



// Image used to display the water appearance
struct field_display
{
    vcl::image_raw im;       // Image storage on CPU
    GLuint texture_id;       // Texture stored on GPU
    vcl::mesh_drawable quad; // Mesh used to display the texture
};

// User parameters available in the GUI
struct gui_parameters
{
    bool display_field;
    bool display_particles;
    bool save_field;
};

struct scene_model : scene_base
{

    void setup_data(std::map<std::string, GLuint> &shaders, scene_structure &scene, gui_structure &gui);
    void frame_draw(std::map<std::string, GLuint> &shaders, scene_structure &scene, gui_structure &gui);
    void display(std::map<std::string, GLuint> &shaders, scene_structure &scene, gui_structure &gui);

    std::vector<particle_element> particles;
    sph_parameters sph_param;

    void update_density();
    void update_pression();
    void update_acceleration();

    std::pair<float, float> evaluate_display_field(const vcl::vec3 &p);

    void initialize_sph();
    void initialize_field_image();
    void set_gui();

    gui_parameters gui_param;
    field_display field_image;
    vcl::mesh_drawable sphere0, sphere1;
    vcl::segments_drawable borders;

    vcl::timer_event timer;

    float kernel(vcl::vec3 p);
    vcl::vec3 grad_kernel(vcl::vec3 p);
};

#endif
