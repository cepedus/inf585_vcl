#pragma once

#include "scenes/base/base.hpp"
#ifdef SCENE_SPH3

#include "vcl/vcl.hpp"


// Store element corresponding to an object
struct shape_matching_object {

    // Create a new shape given
    // - A basic mesh model
    // - Its position
    // - Its speed of the center of mass
    // - The angular speed of the object
    // - Its color
    shape_matching_object(const vcl::mesh& m, const vcl::vec3& center={0,0,0}, const vcl::vec3& speed={0,0,0},const vcl::vec3& angular_speed={0,0,0}, const vcl::vec3& color={1,1,1});
    shape_matching_object();

    vcl::buffer<vcl::vec3> p; // Set of positions describing the shape
    vcl::buffer<vcl::vec3> v; // Associated speed
    vcl::buffer<vcl::vec3> p_save; // Storage of previous position (used to compute the speed before/after the deformation)

    vcl::buffer<vcl::vec3> r0; // Storage of the relative coordinate of each vertex in the reference frame (can be updated in the case of plastic deformation)

    vcl::vec3 COM;  // The Center Of Mass (barycenter of the shape)

    vcl::buffer<vcl::uint3> connectivity; // Connectivity of the surface

    vcl::mesh_drawable visual;       // Visual model to be displayed (updated at each deformation)
    vcl::buffer<vcl::vec3> normals;  // Storage for normals


    void update_center_of_mass(); // recompute the center of mass (based on positions p and connectivity)
    void update_visual_model();   // update the position and normal of the visual model
};

#endif
