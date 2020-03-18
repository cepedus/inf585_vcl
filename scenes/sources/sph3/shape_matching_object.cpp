#include "shape_matching_object.hpp"
#ifdef SCENE_SPH3

using namespace vcl;


shape_matching_object::shape_matching_object() {}
shape_matching_object::shape_matching_object(const vcl::mesh& m, const vcl::vec3& center, const vcl::vec3& speed, const vcl::vec3& angular_speed, const vec3& color)
{
    const size_t N = m.position.size();

    p = m.position;
    for(size_t k=0; k<N; ++k)
        p[k] = p[k] + center;
    p_save = p;

    connectivity = m.connectivity;

    COM = center_of_mass(p, connectivity);
    v.resize(N);
    for(size_t k=0; k<N; ++k)
        v[k] = speed + cross(angular_speed,p[k]-COM);

    r0.resize(N);
    for(size_t k=0; k<N; ++k)
        r0[k] = p[k]-COM;

    visual = m;
    normals = m.normal;
    visual.uniform.color = color;

}

void shape_matching_object::update_center_of_mass(){
    COM = center_of_mass(p, connectivity);
}

void shape_matching_object::update_visual_model(){
    visual.update_position(p);      // update position
    normal(p,connectivity,normals); // recompute new normals
    visual.update_normal(normals);  // update normals
}

#endif
