
#include "sph.hpp"
#include "omp.h"
#ifdef SCENE_SPH1_OPT

using namespace vcl;

// Counter used to save image on hard drive
int counter_image = 0;
const float PI = 3.14159;
const float alpha = 2.f;
const float eps = 1e-12;

int scene_model::hash(vec3 p) {
    float width, height, depth;
    float step = bbox.step;
    int loc;
    width = bbox.width;
    height = bbox.height;
    depth = bbox.depth;
    loc = int(p.x / step) + bbox.nwidth * int(p.y / step) + bbox.nheight*int(p.z / step);
    return loc;
}

void scene_model::initialize_sph()
{
#pragma omp parallel
    printf("thread %d\n", omp_get_thread_num());
    // Influence distance of a particle (size of the kernel)
    const float h = 0.1f;
    bbox.step = h;

    // Rest density (consider 1000 Kg/m^3)
    const float rho0 = 1000.0f;

    // Stiffness (consider ~2000 - used in tait equation)
    const float stiffness = 2000.0f;

    // Viscosity parameter
    const float nu = 2.0f;

    // Total mass of a particle (consider rho0 h^2)
    const float m = rho0 * h * h;

    // Initial particle spacing (relative to h)
    const float c = 0.95f;

    float step = c*h;
    // vec3 mid = (bbox.max - bbox.min) / 2;
    // const float sized = 0.4;
    // vec3 diag{sized,sized,sized};
    // vec3 min = mid - diag;
    // vec3 max = mid + diag;
    const vec3 min{-0.3f, 0.f,0.0f};
    const vec3 max{0.3f, 2.f,step};
    // const vec3 max{0.5f,1.0f,0.475f};
    int loc;

    // Fill a square with particles
    const float epsilon = 1e-3f;
    for (float x = min.x; x < max.x; x = x + step)
    {
        for (float y = min.y; y < max.y; y = y + step)
        {
            for (float z = min.z; z < max.z; z = z + step) {
                particle_element particle;
                particle.p = {x + epsilon * rand_interval(), y, z}; // a zero value in z position will lead to a 2D simulation
                particles.push_back(particle);
                // TODO: init grid
                loc = hash(particle.p);
                if (grid.find(loc) == grid.end()) {
                    grid[loc] = std::vector<particle_element>();
                }
                grid[loc].push_back(particle);
            }

        }
    }

    sph_param.h = h;
    sph_param.rho0 = rho0;
    sph_param.nu = nu;
    sph_param.m = m;
    sph_param.stiffness = stiffness;
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
            if (p.y < bbox.min.y)
            {
                p.y = bbox.min.y + epsilon * rand_interval();
                v.y *= -0.1f;
            }
            if (p.x < bbox.min.x)
            {
                p.x = bbox.min.x + epsilon * rand_interval();
                v.x *= -0.1f;
            }
            if (p.x > bbox.max.x)
            {
                p.x = bbox.max.x - epsilon * rand_interval();
                v.x *= -0.1f;
            }
            if (p.z < bbox.min.z)
            {
                p.z = bbox.min.z + epsilon * rand_interval();
                v.z *= -0.1f;
            }
            if (p.z > bbox.max.z)
            {
                p.z = bbox.max.z - epsilon * rand_interval();
                v.z *= -0.1f;
            }
        }

        grid = std::map<int,std::vector<particle_element>>();
        int loc;
        for (size_t k = 0; k < N; ++k) {
            loc = hash(particles[k].p);
            if (grid.find(loc) == grid.end()) {
                grid[loc] = std::vector<particle_element>();
            }
            grid[loc].push_back(particles[k]);
        }
    }

    display(shaders, scene, gui);
}

float scene_model::kernel(vec3 p)
{
    float h = sph_param.h;
    float r = norm(p);
    if (r > h)
    {
        return 0.0;
    }
    else
    {
        float k = 315 / (64 * PI * pow(h, 3));
        float v = pow(1. - pow(r / h, 2), 3);
        return k * v;
    }
}

vec3 scene_model::grad_kernel(vec3 p)
{
    float h = sph_param.h;
    float r = norm(p);
    if (r > h)
    {
        vec3 vc = {0., 0., 0.};
        return vc;
    }
    else
    {
        float k = -945 / (32 * PI * pow(h, 5));
        float v = pow(1. - pow(r / h, 2), 2);
        return k * v * p;
    }
}

// float scene_model::kernel(vec3 p)
// {
//     float h = sph_param.h;
//     float r = norm(p);
//     float q = r/h;
//     float nconst = (1./PI) / pow(h,3);
//     float tmp;
//     if (r > eps)
//     {
//         if (q > 2.f)
//         {
//             return 0.f;
//         }
//         else if (q > 1.f)
//         {
//             tmp = 0.25f*pow(2.f - q, 3);
//         }
//         else
//         {
//             tmp = 1.f - 1.5f*pow(q,2) + 0.75f*pow(q,3);
//         }
//         return nconst * tmp;
//     } else
//     {
//         return 0.f;
//     }
// }

// vec3 scene_model::grad_kernel(vec3 p)
// {
//     float h = sph_param.h;
//     float r = norm(p);
//     float q = r/h;
//     float nconst = (1./PI) / pow(h,3);
//     float tmp, wdash;
//     if (r > eps)
//     {
//         if (q > 2.f)
//         {
//             wdash = 0.f;
//         }
//         else if (q > 1.f)
//         {
//             wdash = -0.75f*pow(2.f - q, 2);
//         }
//         else
//         {
//             wdash = - 3.f*q*(1.f - 0.75f * q);
//         }
//         wdash = wdash * nconst;
//         tmp = wdash / (r*h);
//     }
//     else
//     {
//         tmp = 0.f;
//     }
//     return tmp * p;
// }

void scene_model::update_density()
{
    const size_t N = particles.size();
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        float rho = 0.0;
        int loc = hash(particles[i].p);
        std::vector<int> neighb = {loc, loc+1, loc-1, loc+bbox.nwidth, loc-bbox.nwidth, loc+bbox.nwidth+1, loc-bbox.nwidth+1, loc+bbox.nwidth-1, loc-bbox.nwidth-1,

        loc-bbox.nheight, loc+1-bbox.nheight, loc-1-bbox.nheight, loc+bbox.nwidth-bbox.nheight, loc-bbox.nwidth-bbox.nheight, loc+bbox.nwidth+1-bbox.nheight, loc-bbox.nwidth+1-bbox.nheight, loc+bbox.nwidth-1-bbox.nheight, loc-bbox.nwidth-1-bbox.nheight,

        loc+bbox.nheight, loc+1+bbox.nheight, loc-1+bbox.nheight, loc+bbox.nwidth+bbox.nheight, loc-bbox.nwidth+bbox.nheight, loc+bbox.nwidth+1+bbox.nheight, loc-bbox.nwidth+1+bbox.nheight, loc+bbox.nwidth-1+bbox.nheight, loc-bbox.nwidth-1+bbox.nheight};

        // std::vector<int> neighb = {loc};
        #pragma omp parallel for reduction(+: rho)
        for(int j = 0; j < neighb.size(); j++) {
            int loc_j = neighb[j];
            if (grid.find(loc_j) != grid.end()) {
                for(auto const& p_j: grid[loc_j]) {
                    vec3 p = particles[i].p - p_j.p;
                    rho += sph_param.m * kernel(p);
                }
            }
        }
        particles[i].rho = rho;
    }
}

void scene_model::update_pression()
{
    const size_t N = particles.size();
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        if (particles[i].rho > sph_param.rho0)
        {
            particles[i].pression =
                sph_param.stiffness * pow(particles[i].rho / sph_param.rho0 - 1.0, alpha);
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
    float h = sph_param.h;

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        particles[i].a = vec3{0, -9.81f, 0};

        // Add contribution of SPH forces

        int loc = hash(particles[i].p);
        std::vector<int> neighb = {loc, loc+1, loc-1, loc+bbox.nwidth, loc-bbox.nwidth, loc+bbox.nwidth+1, loc-bbox.nwidth+1, loc+bbox.nwidth-1, loc-bbox.nwidth-1,

        loc-bbox.nheight, loc+1-bbox.nheight, loc-1-bbox.nheight, loc+bbox.nwidth-bbox.nheight, loc-bbox.nwidth-bbox.nheight, loc+bbox.nwidth+1-bbox.nheight, loc-bbox.nwidth+1-bbox.nheight, loc+bbox.nwidth-1-bbox.nheight, loc-bbox.nwidth-1-bbox.nheight,

        loc+bbox.nheight, loc+1+bbox.nheight, loc-1+bbox.nheight, loc+bbox.nwidth+bbox.nheight, loc-bbox.nwidth+bbox.nheight, loc+bbox.nwidth+1+bbox.nheight, loc-bbox.nwidth+1+bbox.nheight, loc+bbox.nwidth-1+bbox.nheight, loc-bbox.nwidth-1+bbox.nheight};
        // std::vector<int> neighb = {loc};

        #pragma omp parallel for
        for(int neighb_j = 0; neighb_j < neighb.size(); neighb_j++)
        {
            int loc_j = neighb[neighb_j];
            if (grid.find(loc_j) != grid.end())
            {
                size_t M = grid[loc_j].size();
                for(size_t j = 0; j<M; j++)
                {
                    vec3 p_j = grid[loc_j][j].p;
                    float rho_j = grid[loc_j][j].rho;
                    float pression_j = grid[loc_j][j].pression;
                    vec3 v_j = grid[loc_j][j].v;
                    vec3 p = particles[i].p - p_j;

                    if (particles[i].rho > 0.f && rho_j > 0.f) {
                        particles[i].a = particles[i].a - sph_param.m * ((particles[i].pression / pow(particles[i].rho, 2)) + (pression_j / pow(rho_j, 2))) * grad_kernel(p);

                        particles[i].a = particles[i].a + 2 * sph_param.nu * (sph_param.m / rho_j) * (dot(particles[i].p - p_j, particles[i].v - v_j) / (pow(norm(particles[i].p - p_j), 2) + eps * h * h)) * grad_kernel(particles[i].p - p_j);
                    }

                }
            }
        }
    }
}

void scene_model::setup_data(std::map<std::string, GLuint> &shaders, scene_structure &, gui_structure &gui)
{
    gui.show_frame_camera = false;

    sphere = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere.shader = shaders["mesh"];
    sphere.uniform.color = {0, 0.5, 1};

    std::vector<vec3> borders_segments = {
        {bbox.min.x, bbox.min.y, bbox.min.z}, {bbox.max.x, bbox.min.y, bbox.min.z},
        {bbox.min.x, bbox.min.y, bbox.min.z}, {bbox.min.x, bbox.max.y, bbox.min.z},
        {bbox.min.x, bbox.min.y, bbox.min.z}, {bbox.min.x, bbox.min.y, bbox.max.z},
        {bbox.max.x, bbox.min.y, bbox.min.z}, {bbox.max.x, bbox.max.y, bbox.min.z},
        {bbox.max.x, bbox.min.y, bbox.min.z}, {bbox.max.x, bbox.min.y, bbox.max.z},
        {bbox.min.x, bbox.max.y, bbox.min.z}, {bbox.min.x, bbox.max.y, bbox.max.z},
        {bbox.min.x, bbox.max.y, bbox.min.z}, {bbox.max.x, bbox.max.y, bbox.min.z},
        {bbox.min.x, bbox.min.y, bbox.max.z}, {bbox.min.x, bbox.max.y, bbox.max.z},
        {bbox.min.x, bbox.min.y, bbox.max.z}, {bbox.max.x, bbox.min.y, bbox.max.z},
        {bbox.max.x, bbox.max.y, bbox.min.z}, {bbox.max.x, bbox.max.y, bbox.max.z},
        {bbox.min.x, bbox.max.y, bbox.max.z}, {bbox.max.x, bbox.max.y, bbox.max.z},
        {bbox.max.x, bbox.min.y, bbox.max.z}, {bbox.max.x, bbox.max.y, bbox.max.z}
    };
    borders = segments_gpu(borders_segments);
    borders.uniform.color = {0, 0, 0};
    borders.shader = shaders["curve"];

    initialize_sph();
    initialize_field_image();
    sphere.uniform.transform.scaling = sph_param.h / 5.0f;

    gui_param.display_field = true;
    gui_param.display_particles = true;
    gui_param.save_field = true;
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
            sphere.uniform.transform.translation = particles[k].p;
            draw(sphere, scene.camera);
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

                const float f = evaluate_display_field({x, y, 0.0f});
                const float value = 0.5f * std::max(f - 0.25f, 0.0f);

                float r = 1 - value;
                float g = 1 - value;
                float b = 1;

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
}

// Fill an image with field computed as a distance function to the particles
float scene_model::evaluate_display_field(const vcl::vec3 &p)
{
    float field = 0.0f;
    const float d = 0.1f;
    const size_t N = particles.size();
    for (size_t i = 0; i < N; ++i)
    {
        const vec3 &pi = particles[i].p;
        const float r = norm(p - pi);
        const float u = r / d;
        if (u < 4)
            field += std::exp(-u * u);
    }
    return field;
}

// Initialize an image where local density is displayed
void scene_model::initialize_field_image()
{
    size_t N = 50; // Image dimension

    mesh quad = mesh_primitive_quad(
        {bbox.min.x, bbox.min.y, bbox.min.z}, {bbox.max.x, bbox.min.y, bbox.min.z},
        {bbox.max.x, bbox.max.y, bbox.min.z}, {bbox.min.x, bbox.max.y, bbox.min.z});
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

#endif
