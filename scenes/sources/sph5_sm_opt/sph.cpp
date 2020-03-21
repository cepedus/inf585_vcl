
#include "sph.hpp"
#include "omp.h"
#ifdef SCENE_SPH5_SM

using namespace vcl;

// Counter used to save image on hard drive
int counter_image = 0;
const float PI = 3.14159;
const float alpha = 2.0;
const float eps = 1e-3;

int scene_model::hash(vec3 p) {
    float width, height, depth;
    float step = bbox.step;
    int loc;
    width = bbox.width;
    height = bbox.height;
    depth = bbox.depth;
    loc = int(p.x / step) + bbox.nwidth * int(p.y / step) + bbox.nheight * int(p.z / step);
    return loc;
}

void scene_model::initialize_sph()
{
#pragma omp parallel
    printf("thread %d\n", omp_get_thread_num());


    // What do we want to see

    bool two_fluids = false;
    bool baseline = false;
    bool falling_object = true;



    // Influence distance of a particle (size of the kernel)
    const float h = 0.05f;
    bbox.step = h;

    // Rest density (consider 1000 Kg/m^3)
    const float rho0 = 1000.0f;
    const float rho1 = rho0 / 10.0f;

    const float rho_solid = 2 * rho0;

    // Stiffness (consider ~2000 - used in tait equation)
    const float stiffness = 2000.0f;

    // Viscosity parameter
    const float nu = 2.0f;

    // Initial particle spacing (relative to h)
    const float c = 1.0f;

    object_mass = 0.0f;

    // Fill a square with particles
    const float epsilon = 1e-3f;
    for (float x = -1.0f + h; x < 1.0f - h; x = x + c * h)
    {
        for (float y = -1.0f + h; y < 3.0f - h; y = y + c * h)
        {

            // Just one base fluid to see dynamics
            if (baseline)
            {
                if (x > 0.75f && y > 0.0f)
                {
                    particle_element particle;
                    particle.p = { x + epsilon * rand_interval(), y, 0.0f };
                    particle.rho0 = rho0;
                    particle.m = particle.rho0 * h * h;
                    particles.push_back(particle);
                }
            }
            // Two fluids of different density
            else if (two_fluids)
            {
                if (x > -0.3f && x <= 0.3f && y > 1.5f)
                {
                    particle_element particle;
                    particle.p = { x + epsilon * rand_interval(), y, 0.0f };
                    particle.rho0 = rho0;
                    particle.m = particle.rho0 * h * h;
                    particles.push_back(particle);
                }
                else if (x > -0.3f && x <= 0.3f && y > 0.0f && y <= 1.5f)
                {
                    particle_element particle;
                    particle.p = { x + epsilon * rand_interval(), y, 0.0f };
                    particle.rho0 = rho1;
                    particle.m = particle.rho0 * h * h;
                    particles.push_back(particle);
                }
            }

            else if (falling_object)
            {
                if (y < 0.0f)
                {
                    particle_element particle;
                    particle.p = { x + epsilon * rand_interval(), y, 0.0f };
                    particle.rho0 = rho0;
                    particle.m = particle.rho0 * h * h;
                    particles.push_back(particle);
                }
                else if (x > -0.1f && x <= 0.1f && y > 0.3f && y <= 0.6f)
                {
                    particle_element particle;
                    particle.p = { x + epsilon * rand_interval(), y, 0.0f };
                    particle.rho0 = rho_solid;
                    object_i.push_back(particles.size());
                    object_mass += particle.rho0 * h * h;

                    particle.m = particle.rho0 * h * h;
                    particles.push_back(particle);
                }
            }


































            ////for (float z = -1.0f + h; z < 0.5f - h; z = z + c * h)
            ////{
            //    particle_element particle;
            //    particle.p = { x + epsilon * rand_interval(), y, 0.0f }; // a zero value in z position will lead to a 2D simulation
            //    if (y > -0.0f - h) {
            //        particle.rho0 = rho0;
            //    }
            //    else if (x < 0.5f - h) {
            //        particle.rho0 = rho1;
            //    }
            //    else {
            //        particle.rho0 = rho_solid;
            //        object_i.push_back(particles.size());
            //        object_mass += particle.rho0 * h * h;
            //    }
            //    particle.m = particle.rho0 * h * h;
            //    particles.push_back(particle);
            ////}
        }
    }


    is_2d = true;
    sph_param.h = h;
    sph_param.nu = nu;
    sph_param.stiffness = stiffness;
    sph_param.k = 315 / (64 * PI * pow(h, 3));
    sph_param.gradk = -945 / (32 * PI * pow(h, 5));
}

void scene_model::frame_draw(std::map<std::string, GLuint>& shaders, scene_structure& scene, gui_structure& gui)
{
    const float dt = timer.update();
    set_gui();

    // Force constant time step
    float h = dt <= 1e-6f ? 0.0f : timer.scale * 0.0003f;

    const int N_substep = 15;
    for (int k_substep = 0; k_substep < N_substep; ++k_substep)
    {
        // Pre-computations for shape matching
        vec3 COM_0 = vec3(0.0f, 0.0f, 0.0f);
        for (int i : object_i)
        {
            particle_element& p = particles[i];
            COM_0 += p.m * p.p;
        }

        COM_0 /= object_mass;

        std::vector<vec3> r0 = std::vector<vec3>();
        for (int i = 0; i < object_i.size(); i++)
        {
            particle_element& p = particles[object_i[i]];
            r0.push_back(p.p - COM_0);
        }


        // Update values
        update_density();      // First compute updated density
        update_pression();     // Compute associated pression
        update_acceleration(); // Update acceleration

        // Numerical integration
        const float damping = 0.5f;
        const int N = particles.size();
        for (int k = 0; k < N; ++k)
        {
            vec3& p = particles[k].p;
            vec3& v = particles[k].v;
            vec3& a = particles[k].a;

            v = (1 - h * damping) * v + h * a;
            p = p + h * v;
        }

        // Collision
        const float epsilon = 1e-3f;
        for (size_t k = 0; k < N; ++k)
        {
            vec3& p = particles[k].p;
            vec3& v = particles[k].v;

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
        }

        grid = std::map<int, std::vector<particle_element>>();
        int loc;
        for (size_t k = 0; k < N; ++k) {
            loc = hash(particles[k].p);
            if (grid.find(loc) == grid.end()) {
                grid[loc] = std::vector<particle_element>();
            }
            grid[loc].push_back(particles[k]);
        }

        // Force matching after deformation

        vec3 COM = vec3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < object_i.size(); i++)
        {
            particle_element& p = particles[object_i[i]];
            COM += p.m * p.p;
        }

        COM /= object_mass;

        mat3 A = mat3::zero();

        for (int i = 0; i < object_i.size(); i++)
        {
            vec3 r = particles.at(object_i.at(i)).p - COM;
            mat3 A_i = mat3(r.x * r0[i].x, r.x * r0[i].y, r.x * r0[i].z,
                r.y * r0[i].x, r.y * r0[i].y, r.y * r0[i].z,
                r.z * r0[i].x, r.z * r0[i].y, r.z * r0[i].z);
            A += A_i;
        }

        if (is_2d)
        {
            mat2 R = mat2(A(0, 0), A(0, 1), A(1, 0), A(1, 1));

            // Exact polar for 2x2 http://www.cs.cornell.edu/courses/cs4620/2019fa/materials/polarnotes.pdf

            float theta = atan2f(R(1, 0) - R(0, 1), R(0, 0) - R(1, 1));
            R = mat2(cosf(theta), -sinf(theta), sinf(theta), cosf(theta));

            // back to 3d for vector product
            mat3 R_fix = mat3(R(0, 0), R(0, 1), 0.0f, R(1, 0), R(1, 1), 0.0f, 0.0f, 0.0f, 1.0f);

            for (int i = 0; i < object_i.size(); i++)
            {
                particle_element& p = particles.at(object_i.at(i));
                vec3 p0 = p.p;
                p.p = R_fix * r0[i] + COM;
            }
        }
        else
        {
            // from course notes
            mat3 R = A;
            for (int i = 0; i < 10; i++)
                if (det(R) != 0.0f)
                    R = 0.5 * (R + transpose(inverse(R)));

            for (int i = 0; i < object_i.size(); i++)
            {
                particle_element& p = particles.at(object_i.at(i));
                vec3 p0 = p.p;
                p.p = R * r0[i] + COM;
            }
        }

    }

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
        vec3 vc = { 0., 0., 0. };
        return vc;
    }
    else
    {
        return sph_param.gradk * pow(1. - pow(norm(p) / sph_param.h, 2), 2) * p;
    }
}

void scene_model::update_density()
{
    const int N = particles.size();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        float rho = 0.0;
        int loc = hash(particles[i].p);
        std::vector<int> neighb = { loc, loc + 1, loc - 1, loc + bbox.nwidth, loc - bbox.nwidth, loc + bbox.nwidth + 1, loc - bbox.nwidth + 1, loc + bbox.nwidth - 1, loc - bbox.nwidth - 1,

        loc - bbox.nheight, loc + 1 - bbox.nheight, loc - 1 - bbox.nheight, loc + bbox.nwidth - bbox.nheight, loc - bbox.nwidth - bbox.nheight, loc + bbox.nwidth + 1 - bbox.nheight, loc - bbox.nwidth + 1 - bbox.nheight, loc + bbox.nwidth - 1 - bbox.nheight, loc - bbox.nwidth - 1 - bbox.nheight,

        loc + bbox.nheight, loc + 1 + bbox.nheight, loc - 1 + bbox.nheight, loc + bbox.nwidth + bbox.nheight, loc - bbox.nwidth + bbox.nheight, loc + bbox.nwidth + 1 + bbox.nheight, loc - bbox.nwidth + 1 + bbox.nheight, loc + bbox.nwidth - 1 + bbox.nheight, loc - bbox.nwidth - 1 + bbox.nheight };

        // std::vector<int> neighb = {loc};
#pragma omp parallel for reduction(+: rho)
        for (int j = 0; j < neighb.size(); j++) {
            int loc_j = neighb[j];
            if (grid.find(loc_j) != grid.end()) {
                for (auto const& p_j : grid[loc_j]) {
                    vec3 p = particles[i].p - p_j.p;
                    rho += particles[i].m * kernel(p);
                }
            }
        }
        particles[i].rho = rho;
    }
}

void scene_model::update_pression()
{
    // Fill particles[i].pression = ...
    const int N = particles.size();
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
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
    float h = sph_param.h;

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        particles[i].a = vec3{ 0, -9.81f, 0 };

        // Add contribution of SPH forces

        int loc = hash(particles[i].p);
        std::vector<int> neighb = { loc, loc + 1, loc - 1, loc + bbox.nwidth, loc - bbox.nwidth, loc + bbox.nwidth + 1, loc - bbox.nwidth + 1, loc + bbox.nwidth - 1, loc - bbox.nwidth - 1,

        loc - bbox.nheight, loc + 1 - bbox.nheight, loc - 1 - bbox.nheight, loc + bbox.nwidth - bbox.nheight, loc - bbox.nwidth - bbox.nheight, loc + bbox.nwidth + 1 - bbox.nheight, loc - bbox.nwidth + 1 - bbox.nheight, loc + bbox.nwidth - 1 - bbox.nheight, loc - bbox.nwidth - 1 - bbox.nheight,

        loc + bbox.nheight, loc + 1 + bbox.nheight, loc - 1 + bbox.nheight, loc + bbox.nwidth + bbox.nheight, loc - bbox.nwidth + bbox.nheight, loc + bbox.nwidth + 1 + bbox.nheight, loc - bbox.nwidth + 1 + bbox.nheight, loc + bbox.nwidth - 1 + bbox.nheight, loc - bbox.nwidth - 1 + bbox.nheight };
        // std::vector<int> neighb = {loc};

#pragma omp parallel for
        for (int neighb_j = 0; neighb_j < neighb.size(); neighb_j++)
        {
            int loc_j = neighb[neighb_j];
            if (grid.find(loc_j) != grid.end())
            {
                size_t M = grid[loc_j].size();
                for (size_t j = 0; j < M; j++)
                {
                    vec3 p_j = grid[loc_j][j].p;
                    float rho_j = grid[loc_j][j].rho;
                    float pression_j = grid[loc_j][j].pression;
                    vec3 v_j = grid[loc_j][j].v;
                    vec3 p = particles[i].p - p_j;

                    if (particles[i].rho > 0.f && rho_j > 0.f) {
                        particles[i].a = particles[i].a - particles[j].m * ((particles[i].pression / pow(particles[i].rho, 2)) + (pression_j / pow(rho_j, 2))) * grad_kernel(p);

                        particles[i].a = particles[i].a + 2 * sph_param.nu * (particles[j].m / rho_j) * (dot(particles[i].p - p_j, particles[i].v - v_j) / (pow(norm(particles[i].p - p_j), 2) + eps * h * h)) * grad_kernel(particles[i].p - p_j);
                    }

                }
            }
        }
    }
}

void scene_model::setup_data(std::map<std::string, GLuint>& shaders, scene_structure&, gui_structure& gui)
{
    gui.show_frame_camera = false;

    sphere0 = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere0.shader = shaders["mesh"];
    sphere0.uniform.color = { 0, 0.5, 1 };

    //std::vector<vec3> borders_segments = {
    //    {bbox.min.x, bbox.min.y, bbox.min.z}, { bbox.max.x, bbox.min.y, bbox.min.z },
    //    { bbox.min.x, bbox.min.y, bbox.min.z }, { bbox.min.x, bbox.max.y, bbox.min.z },
    //    { bbox.min.x, bbox.min.y, bbox.min.z }, { bbox.min.x, bbox.min.y, bbox.max.z },
    //    { bbox.max.x, bbox.min.y, bbox.min.z }, { bbox.max.x, bbox.max.y, bbox.min.z },
    //    { bbox.max.x, bbox.min.y, bbox.min.z }, { bbox.max.x, bbox.min.y, bbox.max.z },
    //    { bbox.min.x, bbox.max.y, bbox.min.z }, { bbox.min.x, bbox.max.y, bbox.max.z },
    //    { bbox.min.x, bbox.max.y, bbox.min.z }, { bbox.max.x, bbox.max.y, bbox.min.z },
    //    { bbox.min.x, bbox.min.y, bbox.max.z }, { bbox.min.x, bbox.max.y, bbox.max.z },
    //    { bbox.min.x, bbox.min.y, bbox.max.z }, { bbox.max.x, bbox.min.y, bbox.max.z },
    //    { bbox.max.x, bbox.max.y, bbox.min.z }, { bbox.max.x, bbox.max.y, bbox.max.z },
    //    { bbox.min.x, bbox.max.y, bbox.max.z }, { bbox.max.x, bbox.max.y, bbox.max.z },
    //    { bbox.max.x, bbox.min.y, bbox.max.z }, { bbox.max.x, bbox.max.y, bbox.max.z }
    //};

    //borders = segments_gpu(borders_segments);
    //borders.uniform.color = { 0, 0, 0 };
    //borders.shader = shaders["curve"];

    sphere1 = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere1.shader = shaders["mesh"];
    sphere1.uniform.color = { 0.5, 0.0, 0.0 };

    sphere_solid = mesh_drawable(mesh_primitive_sphere(1.0f));
    sphere_solid.shader = shaders["mesh"];
    sphere_solid.uniform.color = { 1.0, 1.0, 1.0 };

    std::vector<vec3> borders_segments = { {-1, -1, -0.1f}, {1, -1, -0.1f}, {1, -1, -0.1f}, {1, 1, -0.1f}, {1, 1, -0.1f}, {-1, 1, -0.1f}, {-1, 1, -0.1f}, {-1, -1, -0.1f}, {-1, -1, 0.1f}, {1, -1, 0.1f}, {1, -1, 0.1f}, {1, 1, 0.1f}, {1, 1, 0.1f}, {-1, 1, 0.1f}, {-1, 1, 0.1f}, {-1, -1, 0.1f}, {-1, -1, -0.1f}, {-1, -1, 0.1f}, {1, -1, -0.1f}, {1, -1, 0.1f}, {1, 1, -0.1f}, {1, 1, 0.1f}, {-1, 1, -0.1f}, {-1, 1, 0.1f} };
    borders = segments_gpu(borders_segments);
    borders.uniform.color = { 0, 0, 0 };
    borders.shader = shaders["curve"];

    initialize_sph();
    initialize_field_image();
    sphere0.uniform.transform.scaling = sph_param.h / 5.0f;
    sphere1.uniform.transform.scaling = sph_param.h / 5.0f;
    sphere_solid.uniform.transform.scaling = sph_param.h / 5.0f;

    gui_param.display_field = false;
    gui_param.display_particles = true;
    gui_param.save_field = false;
}

void scene_model::display(std::map<std::string, GLuint>& shaders, scene_structure& scene, gui_structure&)
{
    draw(borders, scene.camera);

    // Display particles
    if (gui_param.display_particles)
    {
        const int N = particles.size();
        for (int k = 0; k < N; ++k)
        {
            if (particles[k].rho0 == 1000.0f) {
                sphere0.uniform.transform.translation = particles[k].p;
                draw(sphere0, scene.camera);
            }
            else if (particles[k].rho0 == 1000.0f / 10.0f)
            {
                sphere1.uniform.transform.translation = particles[k].p;
                draw(sphere1, scene.camera);
            }
            else { // solid
                sphere_solid.uniform.transform.translation = particles[k].p;
                draw(sphere_solid, scene.camera);
            }
        }
    }

    // Update field image
    if (gui_param.display_field)
    {
        const int im_h = field_image.im.height;
        const int im_w = field_image.im.width;
        std::vector<unsigned char>& im_data = field_image.im.data;
#pragma omp parallel for
        for (int ky = 0; ky < im_h; ++ky)
        {
            for (int kx = 0; kx < im_w; ++kx)
            {
                const float x = 2.0f * kx / (im_w - 1.0f) - 1.0f;
                const float y = 1.0f - 2.0f * ky / (im_h - 1.0f);

                // const float f = evaluate_display_field({x, y, 0.0f});
                const std::pair<float, float> fields = evaluate_display_field({ x, y, 0.0f });
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
}

// Fill an image with field computed as a distance function to the particles
std::pair<float, float> scene_model::evaluate_display_field(const vcl::vec3& p)
{
    float field0 = 0.0f;
    float field1 = 0.0f;
    const float d = 0.1f;
    const int N = particles.size();
#pragma omp parallel for reduction(+ : field0, field1)
    for (int i = 0; i < N; ++i)
    {
        const vec3& pi = particles[i].p;
        const float r = norm(p - pi);
        const float u = r / d;
        if (u < 4) {
            if (particles[i].rho0 == 1000.0f) {
                field0 += std::exp(-u * u);
            }
            else if (particles[i].rho0 == 1000.0f / 10.0f) {
                field1 += std::exp(-u * u);
            }
        }
    }
    std::pair <float, float> res(field0, field1);
    return res;
}

// Initialize an image where local density is displayed
void scene_model::initialize_field_image()
{
    int N = 100; // Image dimension

    mesh quad = mesh_primitive_quad({ -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 });
    quad.texture_uv = { {0, 1}, {1, 1}, {1, 0}, {0, 0} };
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
