
#include "sph.hpp"
#include "omp.h"
#ifdef SCENE_SPH3

using namespace vcl;

// Counter used to save image on hard drive
int counter_image = 0;
const float PI = 3.14159;
const float alpha = 2.0;
const float eps = 0.01;

void scene_model::initialize_sph()
{
#pragma omp parallel
    printf("thread %d\n", omp_get_thread_num());
    // Influence distance of a particle (size of the kernel)
    // const float h = 0.1f/2.f;
    const float h = 0.1f;

    // Rest density (consider 1000 Kg/m^3)
    const float rho0 = 1000.0f;
    const float rho1 = rho0/2.f;

    // Stiffness (consider ~2000 - used in tait equation)
    const float stiffness = 2000.0f;

    // Viscosity parameter
    const float nu = 2.0f;

    // Total mass of a particle (consider rho0 h^2)
    const float m0 = rho0 * h * h;
    const float m1 = rho1 * h * h;

    // Initial particle spacing (relative to h)
    const float c = 0.95f;

    // Fill a square with particles
    const float epsilon = 1e-3f;
    for (float x = h; x < 1.0f - h; x = x + c * h)
    {
        for (float y = -1.0f + h; y < 0.0f - h; y = y + c * h)
        {
            particle_element particle;
            particle.p = {x + epsilon * rand_interval(), y, 0}; // a zero value in z position will lead to a 2D simulation
            particle.rho0 = rho0;
            particle.m = m0;
            particle.rigid = false;
            particle.object = NULL;
            particles.push_back(particle);
        }
    }

    const size_t No = shapes.size();
    for(size_t ko=0; ko<No; ++ko) {
        shape_matching_object& object = shapes[ko];

        for(size_t kv=0; kv<object.p.size(); ++kv) {
            particle_element particle;
            particle.p = object.p[kv];
            particle.rho0 = rho1;
            particle.m = m1;
            particle.rigid = true;
            particle.object = &object;
            particles.push_back(particle);
        }
    }

    sph_param.h = h;
    sph_param.nu = nu;
    sph_param.stiffness = stiffness;
    sph_param.k = 315 / (64 * PI * pow(h, 3));
    sph_param.gradk = -945 / (32 * PI * pow(h, 5));
}

// shape matching
void scene_model::display_shapes_surface(std::map<std::string,GLuint>& shaders, scene_structure& scene)
{
    const size_t No = shapes.size();
    for(size_t ko=0; ko<No; ++ko)
    {
        draw(shapes[ko].visual, scene.camera, shaders.at("mesh_bf"));
        if(gui_param.wireframe)
            draw(shapes[ko].visual, scene.camera, shaders.at("wireframe"));
    }
}

void scene_model::display_bounding_spheres(std::map<std::string,GLuint>& , scene_structure& scene)
{
    const size_t No = shapes.size();
    sphere_visual.uniform.transform.scaling = gui_param.radius_bounding_sphere;
    for(size_t ko=0; ko<No; ++ko) {
        shape_matching_object& object = shapes[ko];

        for(size_t kv=0; kv<object.p.size(); ++kv) {
            sphere_visual.uniform.transform.translation = object.p[kv];
            draw(sphere_visual, scene.camera);
        }
    }

}

void scene_model::frame_draw(std::map<std::string, GLuint> &shaders, scene_structure &scene, gui_structure &gui)
{
    const float dt = timer.update();
    set_gui();

    // Force constant time step
    float h = dt <= 1e-6f ? 0.0f : timer.scale * 0.0003f;

    const size_t N_substep = 15;
    for (size_t k_substep = 0; k_substep < N_substep; ++k_substep)
    {
        // Update values
        update_density();      // First compute updated density
        update_pression();     // Compute associated pression
        update_acceleration(); // Update acceleration

        // Numerical integration
        const float damping = 0.5f;
        const size_t N = particles.size();
        for (size_t k = 0; k < N; ++k)
        {
            vec3 &p = particles[k].p;
            vec3 &v = particles[k].v;
            vec3 &a = particles[k].a;

            v = (1 - h * damping) * v + h * a;
            p = p + h * v;
        }

        // Collision
        const float epsilon = 1e-3f;
        for (size_t k = 0; k < N; ++k)
        {
            vec3 &p = particles[k].p;
            vec3 &v = particles[k].v;

            // small perturbation to avoid alignment
            if (p.y < -1)
            {
                p.y = -1 + epsilon * rand_interval();
                v.y *= -0.1f;
            }
            if (p.x < -1)
            {
                p.x = -1 + epsilon * rand_interval();
                v.x *= -0.1f;
            }
            if (p.x > 1)
            {
                p.x = 1 - epsilon * rand_interval();
                v.x *= -0.1f;
            }
            if (p.z < -0.1)
            {
                p.z = -0.1f + epsilon * rand_interval();
                v.z *= -0.1f;
            }
            if (p.z > 0.1)
            {
                p.z = 0.1f - epsilon * rand_interval();
                v.z *= -0.1f;
            }
        }
    }

    // Display shapes
    display_shapes_surface(shaders, scene);
    if(gui_param.bounding_spheres)
        display_bounding_spheres(shaders, scene);
    display(shaders, scene, gui);
}

float scene_model::kernel(vec3 p)
{
    float h = sph_param.h;
    if (norm(p) > h)
    {
        return 0.0;
    }
    else
    {
        return sph_param.k * pow(1. - pow(norm(p) / sph_param.h, 2), 3);
    }
}

vec3 scene_model::grad_kernel(vec3 p)
{
    float h = sph_param.h;
    if (norm(p) > h)
    {
        vec3 vc = {0., 0., 0.};
        return vc;
    }
    else
    {
        return sph_param.gradk * pow(1. - pow(norm(p) / sph_param.h, 2), 2) * p;
    }
}

void scene_model::update_density()
{
    // Fill particles[i].rho = ...
    const size_t N = particles.size();
    vec3 p;
    float rho;
    // #pragma omp parallel for shared(rho, particles, p, j) ordered
    for (size_t i = 0; i < N; i++)
    {
        rho = 0.0;
        #pragma omp parallel for reduction(+: rho)
        for (size_t j = 0; j < N; j++)
        {
            if (false) {

            } else {
                p = particles[i].p - particles[j].p;
                rho += particles[j].m * kernel(p);
            }
        }
        particles[i].rho = rho;
    }
}

void scene_model::update_pression()
{
    // Fill particles[i].pression = ...
    const size_t N = particles.size();
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        if (particles[i].rho > (particles[i].rho0))
        {
            particles[i].pression =
                sph_param.stiffness * pow(particles[i].rho / particles[i].rho0 - 1.0, alpha);
        }
        else
        {
            particles[i].pression = 0.0;
        }
    }
}

void scene_model::update_acceleration()
{
    // gravity
    const size_t N = particles.size();
    vec3 p;
    float h = sph_param.h;
    // #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        particles[i].a = vec3{0, -9.81f, 0};

        // Add contribution of SPH forces
        // particles[i].a += ... (contribution from pression and viscosity)
        #pragma omp parallel for shared(p)
        for (size_t j = 0; j < N; ++j)
        {
            if (false) {

            } else {
                p = particles[i].p - particles[j].p;

                particles[i].a = particles[i].a - particles[j].m * ((particles[i].pression / pow(particles[i].rho, 2)) + (particles[j].pression / pow(particles[j].rho, 2))) * grad_kernel(p);

                particles[i].a = particles[i].a + 2 * sph_param.nu * (particles[j].m / particles[j].rho) * (dot(particles[i].p - particles[j].p, particles[i].v - particles[j].v) / (pow(norm(particles[i].p - particles[j].p), 2) + eps * h * h)) * grad_kernel(particles[i].p - particles[j].p);
            }
        }

    }
}

void scene_model::setup_data(std::map<std::string, GLuint> &shaders, scene_structure &, gui_structure &gui)
{
    gui.show_frame_camera = false;

    // Initialize drawable parameters for shape matching
    gui_param.radius_bounding_sphere = 0.2f;
    object_length = 0.25f;
    sphere_visual = mesh_primitive_sphere(gui_param.radius_bounding_sphere);
    sphere_visual.shader = shaders["mesh"];

    sphere0 = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere0.shader = shaders["mesh"];
    sphere0.uniform.color = {0, 0.5, 1};

    sphere1 = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere1.shader = shaders["mesh"];
    sphere1.uniform.color = {0.5, 0.0, 0.0};

    std::vector<vec3> borders_segments = {{-1, -1, -0.1f}, {1, -1, -0.1f}, {1, -1, -0.1f}, {1, 1, -0.1f}, {1, 1, -0.1f}, {-1, 1, -0.1f}, {-1, 1, -0.1f}, {-1, -1, -0.1f}, {-1, -1, 0.1f}, {1, -1, 0.1f}, {1, -1, 0.1f}, {1, 1, 0.1f}, {1, 1, 0.1f}, {-1, 1, 0.1f}, {-1, 1, 0.1f}, {-1, -1, 0.1f}, {-1, -1, -0.1f}, {-1, -1, 0.1f}, {1, -1, -0.1f}, {1, -1, 0.1f}, {1, 1, -0.1f}, {1, 1, 0.1f}, {-1, 1, -0.1f}, {-1, 1, 0.1f}};
    borders = segments_gpu(borders_segments);
    borders.uniform.color = {0, 0, 0};
    borders.shader = shaders["curve"];

    // Create the basic mesh models: Cube, Cylinder, Torus
    mesh_basic_model[surface_cube]     = mesh_primitive_bar_grid(4,4,2,{0,0,0},{object_length,0,0},{0,object_length,0},{0,0,object_length/4});

    // Initialize a initial scene
    initialize_shapes();

    initialize_sph();
    initialize_field_image();



    // Load specific shaders
    shaders["wireframe_quads"] = create_shader_program("scenes/shared_assets/shaders/wireframe_quads/shader.vert.glsl","scenes/shared_assets/shaders/wireframe_quads/shader.geom.glsl","scenes/shared_assets/shaders/wireframe_quads/shader.frag.glsl"); // Shader for quad meshes
    shaders["normals"] = create_shader_program("scenes/shared_assets/shaders/normals/shader.vert.glsl","scenes/shared_assets/shaders/normals/shader.geom.glsl","scenes/shared_assets/shaders/normals/shader.frag.glsl"); // Shader to display normals

    sphere0.uniform.transform.scaling = sph_param.h / 5.0f;
    sphere1.uniform.transform.scaling = sph_param.h / 5.0f;

    gui_param.display_field = true;
    gui_param.display_particles = true;
    gui_param.save_field = false;
    gui_param.bounding_spheres = true;
    gui_param.wireframe = true;
}

void scene_model::display(std::map<std::string, GLuint> &shaders, scene_structure &scene, gui_structure &)
{
    draw(borders, scene.camera);

    // Display particles
    if (gui_param.display_particles)
    {
        const size_t N = particles.size();
        for (size_t k = 0; k < N; ++k)
        {
            if (particles[k].rho0 == 1000.0f) {
                sphere0.uniform.transform.translation = particles[k].p;
                draw(sphere0, scene.camera);
            } else{
                sphere1.uniform.transform.translation = particles[k].p;
                draw(sphere1, scene.camera);
            }
        }
    }

    // Update field image
    if (gui_param.display_field)
    {
        const size_t im_h = field_image.im.height;
        const size_t im_w = field_image.im.width;
        std::vector<unsigned char> &im_data = field_image.im.data;
        #pragma omp parallel for
        for (size_t ky = 0; ky < im_h; ++ky)
        {
            for (size_t kx = 0; kx < im_w; ++kx)
            {
                const float x = 2.0f * kx / (im_w - 1.0f) - 1.0f;
                const float y = 1.0f - 2.0f * ky / (im_h - 1.0f);

                // const float f = evaluate_display_field({x, y, 0.0f});
                const std::pair<float, float> fields = evaluate_display_field({x, y, 0.0f});
                const float valuef1 = 0.5f * std::max(fields.first - 0.25f, 0.0f);
                const float valuef2 = 0.5f * std::max(fields.second - 0.25f, 0.0f);


                float r;
                float g;
                float b;
                r = 1 - valuef1;
                g = 1 - valuef2;
                b = 1;

                im_data[4 * (kx + im_w * ky)] = static_cast<unsigned char>(255 * std::max(std::min(r, 1.0f), 0.0f));
                im_data[4 * (kx + im_w * ky) + 1] = static_cast<unsigned char>(255 * std::max(std::min(g, 1.0f), 0.0f));
                im_data[4 * (kx + im_w * ky) + 2] = static_cast<unsigned char>(255 * std::max(std::min(b, 1.0f), 0.0f));
                im_data[4 * (kx + im_w * ky) + 3] = 255;
            }
        }

        // Display texture
        glBindTexture(GL_TEXTURE_2D, field_image.texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(im_w), GLsizei(im_h), GL_RGBA, GL_UNSIGNED_BYTE, &im_data[0]);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        draw(field_image.quad, scene.camera, shaders["mesh"]);
        glBindTexture(GL_TEXTURE_2D, scene.texture_white);

        // Save texture on hard drive
        if (gui_param.save_field)
        {
            const std::string filename = vcl::zero_fill(std::to_string(counter_image), 3);
            image_save_png("output/sph/file_" + filename + ".png", field_image.im);
            ++counter_image;
        }
    }
}

void scene_model::set_gui()
{
    // Can set the speed of the animation
    float scale_min = 0.05f;
    float scale_max = 2.0f;
    ImGui::SliderScalar("Time scale", ImGuiDataType_Float, &timer.scale, &scale_min, &scale_max, "%.2f s");

    ImGui::Checkbox("Display field", &gui_param.display_field);
    ImGui::Checkbox("Display particles", &gui_param.display_particles);
    ImGui::Checkbox("Save field on disk", &gui_param.save_field);

    // Start and stop animation
    if (ImGui::Button("Stop"))
        timer.stop();
    if (ImGui::Button("Start"))
        timer.start();

    ImGui::Checkbox("Sphere", &gui_param.bounding_spheres); ImGui::SameLine();
}

// Fill an image with field computed as a distance function to the particles
std::pair<float, float> scene_model::evaluate_display_field(const vcl::vec3 &p)
{
    float field0 = 0.0f;
    float field1 = 0.0f;
    const float d = 0.1f;
    const size_t N = particles.size();
    #pragma omp parallel for reduction(+ : field0, field1)
    for (size_t i = 0; i < N; ++i)
    {
        const vec3 &pi = particles[i].p;
        const float r = norm(p - pi);
        const float u = r / d;
        if (u < 4) {
            if (particles[i].rho0 == 1000.0f) {
                field0 += std::exp(-u * u);
            } else {
                field1 += std::exp(-u * u);
            }
        }
    }
    std::pair <float,float> res (field0, field1);
    return res;
}

// Initialize an image where local density is displayed
void scene_model::initialize_field_image()
{
    size_t N = 100; // Image dimension

    mesh quad = mesh_primitive_quad({-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0});
    quad.texture_uv = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    field_image.quad = quad;

    field_image.im.width = N;
    field_image.im.height = N;
    field_image.im.data.resize(4 * field_image.im.width * field_image.im.height);
    field_image.texture_id = create_texture_gpu(field_image.im);
    field_image.im.color_type = image_color_type::rgba;

    field_image.quad.uniform.shading.ambiant = 1.0f;
    field_image.quad.uniform.shading.diffuse = 0.0f;
    field_image.quad.uniform.shading.specular = 0.0f;
}

void scene_model::initialize_shapes()
{
    // Clear in case there is existing shapes
    shapes.clear();

    // Shorthand notation to setup the basic shape models
    const float& a = object_length;                    // size of the object
    const float& r = gui_param.radius_bounding_sphere; // radius of the bounding sphere

    // Create the scene
    const vec3& color = {1,1,0.7f};
    shapes.push_back( shape_matching_object(mesh_basic_model[surface_cube], {0,0,0}, {0,0,0}, {0,0,0}, color ) );
    shapes[0].update_visual_model();
    // shapes.push_back( shape_matching_object(mesh_basic_model[surface_cube], {a+2*r,0,-1+a/2+r}, {0,0,0}, {0,0,0},color ) );
    // shapes.push_back( shape_matching_object(mesh_basic_model[surface_cube], {-a-2*r,0,-1+a/2+r}, {0,0,0}, {0,0,0},color ) );
    // shapes.push_back( shape_matching_object(mesh_basic_model[surface_cube], {0,0,1} ,{0,0,0} , {0,0,0}, color ) );
    // shapes.push_back( shape_matching_object(mesh_basic_model[surface_cylinder], {0,1,1.5} ,{0,0,0} , {0,0,0}, color ) );
    // shapes.push_back( shape_matching_object(mesh_basic_model[surface_torus], {0,2,1} ,{0,0,0} , {0,0,0}, color ) );
}

#endif
